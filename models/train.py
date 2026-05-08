"""
改进训练脚本：支持GPU加速、混合精度、tf.data管道、XLA优化、Mixup增强。
修复：TF 2.13混合精度API路径，LossScaleOptimizer在外部创建。
新增：Mixup数据增强（样本间混合），提升小数据集泛化能力。
"""
from utils.project_init import set_project_root

set_project_root()

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import config
from data.dataset import ESC50Dataset
from models.lightweight_model import create_tiny_v2
from features.mel_spectrogram import MelSpectrogram
from utils.gpu_utils import setup_gpu_environment, print_gpu_memory_usage

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# ========== 改进：初始化GPU环境 ==========
strategy = setup_gpu_environment()

# 根据特征提取器更新输入形状
tmp_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, tmp_mel.eff_n_mels, 1)

print(f"模型输入形状: {config.MODEL_INPUT_SHAPE}")
print(f"批次大小: {config.BATCH_SIZE}")
print_gpu_memory_usage()


class CosineAnnealingScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    余弦退火学习率调度器，支持Warmup阶段。

    调度策略：
        1. Warmup阶段（前N步）：学习率从0线性增长到max_lr
        2. 退火阶段：余弦曲线从max_lr下降到min_lr

    Attributes:
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: Warmup步数
        total_steps: 总训练步数
    """

    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

    def __call__(self, step):
        """
        根据当前步数计算学习率。

        Args:
            step: 当前训练步数（标量）

        Returns:
            tf.Tensor: 当前步的学习率
        """
        step = tf.cast(step, tf.float32)

        # 线性热身：前warmup_steps步从0增长到max_lr
        warmup_lr = self.max_lr * (step / tf.cast(self.warmup_steps, tf.float32))

        # 余弦退火：按余弦曲线下降
        progress = (step - self.warmup_steps) / tf.cast(
            self.total_steps - self.warmup_steps, tf.float32
        )
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * progress)
        )

        # 步数小于warmup_steps时用warmup_lr，否则用cosine_lr
        return tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup_lr, cosine_lr)


def load_teacher_model():
    """
    加载预训练教师模型（如AST或PANNs的微调版本）。

    Returns:
        tf.keras.Model or None: 教师模型，不存在则返回None
    """
    if not os.path.exists(config.TEACHER_MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.TEACHER_MODEL_PATH)


def distillation_loss(y_true, y_pred_student, y_pred_teacher, T=2.0, alpha=0.7):
    """
    知识蒸馏损失：软标签KL散度 + 硬标签交叉熵。

    软标签通过温度T软化概率分布，传递更多类别间关系信息。

    Args:
        y_true: 真实one-hot标签，形状(batch, num_classes)
        y_pred_student: 学生模型预测概率，形状(batch, num_classes)
        y_pred_teacher: 教师模型预测logits，形状(batch, num_classes)
        T: 蒸馏温度，越高概率分布越平滑
        alpha: 软标签损失权重，(1-alpha)为硬标签权重

    Returns:
        tf.Tensor: 标量损失值
    """
    # 硬标签交叉熵损失
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)

    # 软标签KL散度损失（温度缩放）
    soft_teacher = tf.nn.softmax(y_pred_teacher / T)
    soft_student = tf.nn.softmax(y_pred_student / T)
    kd = tf.keras.losses.KLDivergence()(soft_teacher, soft_student) * (T ** 2)

    return alpha * kd + (1 - alpha) * ce


def create_train_step(model, optimizer, use_mixed_precision=False, teacher=None):
    # Mixup 启用时关闭标签平滑（软标签已有正则效果）
    if config.USE_MIXUP:
        smoothing = 0.0
    else:
        smoothing = config.LABEL_SMOOTHING

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=smoothing
    )

    @tf.function
    def train_step(X, y_true):
        # ---------- 概率性 Mixup ----------
        if config.USE_MIXUP and config.MIXUP_ALPHA > 0:
            # 以 50% 的概率对当前 batch 执行 Mixup，保留纯净样本
            if tf.random.uniform([]) < 0.5:
                # 从 Beta 分布采样 λ（使用 Gamma 近似）
                alpha = tf.constant(config.MIXUP_ALPHA, dtype=tf.float32)
                gamma1 = tf.random.gamma([], alpha, dtype=tf.float32)
                gamma2 = tf.random.gamma([], alpha, dtype=tf.float32)
                lam = gamma1 / (gamma1 + gamma2)

                # 随机打乱 batch 内样本索引
                batch_size = tf.shape(X)[0]
                index = tf.random.shuffle(tf.range(batch_size))

                # 混合输入和标签
                X_mix = lam * X + (1.0 - lam) * tf.gather(X, index)
                y_mix = lam * y_true + (1.0 - lam) * tf.gather(y_true, index)
            else:
                X_mix = X
                y_mix = y_true
        else:
            X_mix = X
            y_mix = y_true

        with tf.GradientTape() as tape:
            preds = model(X_mix, training=True)
            if teacher is not None:
                teacher_logits = teacher(X_mix, training=False)
                loss = distillation_loss(
                    y_mix, preds, teacher_logits,
                    T=config.KD_TEMPERATURE, alpha=config.KD_ALPHA
                )
            else:
                loss = loss_fn(y_mix, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.reduce_mean(loss)

    return train_step


def train_fold(fold, teacher):
    """
    训练单个交叉验证折。

    使用tf.data.Dataset实现数据预取和并行加载，
    配合@tf.function编译的训练步骤最大化GPU利用率。

    Args:
        fold: 验证折编号 (0-4)
        teacher: 教师模型（知识蒸馏用，可为None）

    Returns:
        tuple: (最佳验证准确率, 最终验证准确率, 训练好的模型)
    """
    print(f"\n{'=' * 50}")
    print(f"训练第 {fold + 1}/{config.N_FOLDS} 折")
    print(f"{'=' * 50}")

    # 创建数据集（tf.data管道自动处理预取和并行）
    train_dataset = ESC50Dataset(fold=fold, mode='train', augment=True)
    val_dataset = ESC50Dataset(fold=fold, mode='val', augment=False)

    train_ds = train_dataset.dataset
    val_ds = val_dataset.dataset

    # 计算每epoch步数
    train_samples = len(train_dataset.samples)
    val_samples = len(val_dataset.samples)
    steps_per_epoch = max(1, train_samples // config.BATCH_SIZE)
    validation_steps = max(1, val_samples // config.BATCH_SIZE)

    print(f"训练样本: {train_samples}, 验证样本: {val_samples}")
    print(f"每epoch步数: {steps_per_epoch}, 验证步数: {validation_steps}")

    # 在策略作用域内创建模型和优化器（支持多GPU）
    with strategy.scope():
        model = create_tiny_v2()

        # 基础优化器
        base_optimizer = tf.keras.optimizers.Adam(
            learning_rate=CosineAnnealingScheduler(
                config.MAX_LEARNING_RATE,
                config.MIN_LEARNING_RATE,
                config.WARMUP_EPOCHS,
                config.EPOCHS,
                steps_per_epoch
            )
        )

        # 改进：混合精度优化器在外部创建，使用正确API路径
        if config.MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            print(f"已包装混合精度优化器: {optimizer}")
            use_mixed_precision = True
        else:
            optimizer = base_optimizer
            use_mixed_precision = False

    # 创建编译后的训练步骤函数（内部已包含Mixup）
    train_step_fn = create_train_step(
        model, optimizer,
        use_mixed_precision=use_mixed_precision,
        teacher=teacher
    )

    # 训练循环
    best_val_acc = 0.0
    fold_h5 = f"models/saved/fold_{fold}_best.h5"

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        # ========== 训练阶段 ==========
        train_loss = 0.0
        train_batches = 0

        # 迭代tf.data.Dataset（自动预取下一batch到GPU）
        for step, (X_batch, y_batch) in enumerate(train_ds.take(steps_per_epoch)):
            # 执行训练步骤（@tf.function编译，GPU执行，内部包含Mixup）
            loss = train_step_fn(X_batch, y_batch)
            train_loss += float(loss)
            train_batches += 1

            # 每20步打印进度
            if step % 20 == 0:
                print(f"  Epoch {epoch + 1}/{config.EPOCHS} | "
                      f"Step {step}/{steps_per_epoch} | "
                      f"Loss: {float(loss):.4f}")

        avg_train_loss = train_loss / max(train_batches, 1)

        # ========== 验证阶段 ==========
        val_correct = 0
        val_total = 0

        for X_val, y_val in val_ds.take(validation_steps):
            # 验证时不计算梯度，推理模式
            probs = model(X_val, training=False)
            y_pred = tf.argmax(probs, axis=1)
            y_true = tf.argmax(y_val, axis=1)
            val_correct += tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.int32))
            val_total += tf.shape(y_val)[0]

        val_acc = float(val_correct) / float(val_total)

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} 完成 | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(fold_h5)
            print(f"  -> 保存最佳模型 (acc={val_acc:.4f})")

    # 加载最佳模型进行最终评估
    print(f"  加载最佳模型进行最终评估...")
    model.load_weights(fold_h5)

    # 全验证集评估（提取所有样本特征）
    all_val_data = val_dataset.get_fold_data(fold)
    X_val_all, y_val_all = _extract_all(all_val_data, val_dataset.mel_extractor)

    # 批量预测
    y_pred_all = np.argmax(
        model.predict(X_val_all, verbose=0, batch_size=config.BATCH_SIZE),
        axis=1
    )
    y_true_all = np.argmax(y_val_all, axis=1)
    final_acc = accuracy_score(y_true_all, y_pred_all)

    print(f"  折 {fold} 最终验证准确率: {final_acc:.4f}")
    print_gpu_memory_usage()

    return best_val_acc, final_acc, model


def _extract_all(samples, mel_extractor):
    """
    从样本列表批量提取固定形状特征（用于最终评估）。

    验证集评估时使用，一次性提取所有样本避免迭代开销。

    Args:
        samples: list of (audio_segment, label)，audio为float32[-1,1]
        mel_extractor: MelSpectrogram实例

    Returns:
        tuple: (X, y) numpy数组
            X: 形状(N, 49, 32, 1) float32
            y: 形状(N, num_classes) float32 one-hot
    """
    X, y = [], []
    target_frames = config.MODEL_INPUT_SHAPE[0]  # 49帧

    for audio, label in samples:
        # audio已是float32[-1,1]，转int16给extractor
        audio_int16 = (audio * 32767).astype(np.int16)
        mel = mel_extractor.compute(audio_int16)

        # 填充或截断到固定帧数
        if mel.shape[0] < target_frames:
            pad = np.zeros(
                (target_frames - mel.shape[0], mel.shape[1]),
                dtype=np.float32
            )
            mel = np.vstack([mel, pad])
        else:
            mel = mel[:target_frames]

        X.append(mel)
        y.append(label)

    # 添加通道维度并one-hot编码
    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, 49, 32, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=config.NUM_CLASSES)

    return X, y


def main():
    """主训练流程：五折交叉验证 + 模型导出。"""
    print("=" * 60)
    print("音频事件检测模型训练（GPU加速版 - 集成Mixup增强）")
    print("=" * 60)

    # 加载教师模型（如启用知识蒸馏）
    teacher = None
    if config.USE_KD:
        teacher = load_teacher_model()
        if teacher is None:
            print("警告：未找到教师模型，将不使用知识蒸馏")

    # 五折交叉验证
    fold_accs = []
    for fold in range(config.N_FOLDS):
        best_val, final_acc, model = train_fold(fold, teacher)
        fold_accs.append(final_acc)

    # 汇总结果
    print(f"\n{'=' * 60}")
    print("交叉验证完成")
    print(f"各折准确率: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"平均准确率: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # 选择最佳折并保存最终模型
    best_fold = np.argmax(fold_accs)
    best_h5 = f"models/saved/fold_{best_fold}_best.h5"

    print(f"\n选择折 {best_fold} 作为最佳模型")
    model = tf.keras.models.load_model(best_h5)
    model.save(config.MODEL_SAVE_PATH)
    print(f"模型已保存: {config.MODEL_SAVE_PATH}")

    # 转换为TFLite格式（端侧部署用）
    print("转换为TFLite格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 允许 TensorFlow Select 算子（支持 GRU 等需要 TensorArray 的层）
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # 禁用 TensorList 的静态形状要求
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(config.TFLITE_MODEL_PATH) / 1024
    print(f"TFLite模型已保存: {config.TFLITE_MODEL_PATH}")
    print(f"模型大小: {size_kb:.1f} KB")
    print_gpu_memory_usage()
    print("=" * 60)


if __name__ == "__main__":
    main()
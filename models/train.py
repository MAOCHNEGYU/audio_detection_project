"""
改进训练脚本：支持知识蒸馏、Mixup、余弦退火、五折交叉验证。
"""
from utils.project_init import set_project_root
set_project_root()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import config
from data.dataset import ESC50Dataset
from models.lightweight_model import create_tiny_v2
from features.mel_spectrogram import MelSpectrogram


# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 根据特征提取器更新输入形状
tmp_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, tmp_mel.eff_n_mels, 1)


class CosineAnnealingScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # 线性热身部分
        warmup_lr = self.max_lr * (step / tf.cast(self.warmup_steps, tf.float32))

        # 余弦退火部分
        progress = (step - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * progress))

        # 根据步数选择学习率
        lr = tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup_lr, cosine_lr)
        return lr

def load_teacher_model():
    """加载预训练教师模型（如 AST 或 PANNs 的微调版本）"""
    if not os.path.exists(config.TEACHER_MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.TEACHER_MODEL_PATH)


def distillation_loss(y_true, y_pred_student, y_pred_teacher, T=2.0, alpha=0.7):
    """
    知识蒸馏损失：软标签 KL 散度 + 硬标签交叉熵。
    """
    # 硬标签损失
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred_student)
    # 软标签损失
    soft_teacher = tf.nn.softmax(y_pred_teacher / T)
    soft_student = tf.nn.softmax(y_pred_student / T)
    kd = tf.keras.losses.KLDivergence()(soft_teacher, soft_student) * (T**2)
    return alpha * kd + (1 - alpha) * ce


def train_fold(fold, teacher):
    """训练单个折"""
    train_gen = ESC50Dataset(fold=fold, mode='train', augment=True)
    val_gen = ESC50Dataset(fold=fold, mode='val', augment=False)
    train_samples = len(train_gen.get_fold_data(fold))
    val_samples = len(val_gen.get_fold_data(fold))
    steps_per_epoch = max(1, train_samples // config.BATCH_SIZE)
    validation_steps = max(1, val_samples // config.BATCH_SIZE)

    model = create_tiny_v2()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CosineAnnealingScheduler(
            config.MAX_LEARNING_RATE,
            config.MIN_LEARNING_RATE,
            config.WARMUP_EPOCHS,
            config.EPOCHS,
            steps_per_epoch
        )
    )

    # 自定义训练步骤（支持 Mixup 和 知识蒸馏）
    @tf.function
    def train_step(X, y_true):
        # Mixup
        if config.USE_MIXUP and config.MIXUP_ALPHA > 0:
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            index = tf.random.shuffle(tf.range(tf.shape(X)[0]))
            X_mix = lam * X + (1 - lam) * tf.gather(X, index)
            y_mix = lam * y_true + (1 - lam) * tf.gather(y_true, index)
        else:
            X_mix, y_mix = X, y_true

        with tf.GradientTape() as tape:
            preds = model(X_mix, training=True)
            # 标签平滑
            smooth_y = y_mix * (1 - config.LABEL_SMOOTHING) + config.LABEL_SMOOTHING / config.NUM_CLASSES
            if teacher is not None:
                # 获取教师预测（不需梯度）
                teacher_logits = teacher(X, training=False)
                loss = distillation_loss(smooth_y, preds, teacher_logits, T=config.KD_TEMPERATURE, alpha=config.KD_ALPHA)
            else:
                loss = tf.keras.losses.categorical_crossentropy(smooth_y, preds)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # 训练循环
    best_val_acc = 0
    fold_h5 = f"models/saved/fold_{fold}_best.h5"
    for epoch in range(config.EPOCHS):
        # 训练
        train_gen_iter = train_gen.generator(config.BATCH_SIZE)
        for step in range(steps_per_epoch):
            X_batch, y_batch = next(train_gen_iter)
            # y_batch 是 one-hot
            loss = train_step(X_batch, y_batch)
            if step % 20 == 0:
                tf.print(f"Epoch {epoch+1}/{config.EPOCHS}, Step {step}/{steps_per_epoch}, Loss: {loss}")

        # 验证
        val_acc = 0.0
        val_gen_iter = val_gen.generator(config.BATCH_SIZE)
        num_val_batches = 0
        for _ in range(validation_steps):
            X_val, y_val = next(val_gen_iter)
            probs = model(X_val, training=False)
            y_pred = tf.argmax(probs, axis=1)
            y_true = tf.argmax(y_val, axis=1)
            val_acc += tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
            num_val_batches += 1
        val_acc /= max(num_val_batches, 1)
        tf.print(f"Epoch {epoch+1}/{config.EPOCHS}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(fold_h5)

    # 全验证集最终评估
    model.load_weights(fold_h5)
    all_val_data = val_gen.get_fold_data(fold)
    X_val_all, y_val_all = _extract_all(all_val_data, val_gen.mel_extractor)
    y_pred_all = np.argmax(model.predict(X_val_all, verbose=0), axis=1)
    y_true_all = np.argmax(y_val_all, axis=1)
    final_acc = accuracy_score(y_true_all, y_pred_all)
    return best_val_acc.numpy(), final_acc, model


def _extract_all(samples, mel_extractor):
    """同上，从样本列表批量提取固定形状特征"""
    X, y = [], []
    target_frames = config.MODEL_INPUT_SHAPE[0]
    for audio, label in samples:
        mel = mel_extractor.compute((audio * 32768).astype(np.int16))
        if mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - mel.shape[0], mel.shape[1]), dtype=np.float32)
            mel = np.vstack([mel, pad])
        else:
            mel = mel[:target_frames]
        X.append(mel)
        y.append(label)
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = tf.keras.utils.to_categorical(y, num_classes=config.NUM_CLASSES)
    return X, y


def main():
    teacher = load_teacher_model() if config.USE_KD else None
    if teacher is None and config.USE_KD:
        print("警告：未找到教师模型，将不使用知识蒸馏")

    fold_accs = []
    for fold in range(config.N_FOLDS):
        print(f"\n===== 训练第 {fold+1}/{config.N_FOLDS} 折 =====")
        best_val, final_acc, model = train_fold(fold, teacher)
        fold_accs.append(final_acc)
        print(f"折 {fold} 验证准确率: {final_acc:.4f}")

    print(f"\n===== 交叉验证完成 =====")
    print(f"平均准确率: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # 保存最佳模型并转换 TFLite
    best_fold = np.argmax(fold_accs)
    best_h5 = f"models/saved/fold_{best_fold}_best.h5"
    model = tf.keras.models.load_model(best_h5)
    model.save(config.MODEL_SAVE_PATH)
    # TFLite 转换（FP32，量化后续单独处理）
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite 模型已保存，大小: {os.path.getsize(config.TFLITE_MODEL_PATH)/1024:.1f} KB")


if __name__ == "__main__":
    main()
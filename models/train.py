"""
训练脚本：多分类（含 other 类），在线动态增强数据，纯 CNN 模型，
Mixup 增强，余弦退火，混合精度，TFLite 导出。
"""
from utils.project_init import set_project_root
set_project_root()

import os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
from data.dataset import ESC50Dataset
from models.lightweight_model import create_tiny_model
from features.mel_spectrogram import MelSpectrogram
from utils.gpu_utils import setup_gpu_environment, print_gpu_memory_usage
from utils.metrics import compute_metrics

tf.random.set_seed(42)
np.random.seed(42)

strategy = setup_gpu_environment()
tmp_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, tmp_mel.eff_n_mels, 1)
print(f"模型输入形状: {config.MODEL_INPUT_SHAPE}")

class CosineAnnealingScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.max_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        progress = (step - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * progress))
        return tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup_lr, cosine_lr)

def create_train_step(model, optimizer, use_mixed_precision=False):
    if config.USE_MIXUP:
        smoothing = 0.0
    else:
        smoothing = config.LABEL_SMOOTHING
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=smoothing)

    @tf.function
    def train_step(X, y_true):
        y_true = tf.cast(y_true, tf.float32)
        if config.USE_MIXUP and config.MIXUP_ALPHA > 0:
            if tf.random.uniform([]) < 0.5:
                alpha = tf.constant(config.MIXUP_ALPHA, dtype=tf.float32)
                gamma1 = tf.random.gamma([], alpha)
                gamma2 = tf.random.gamma([], alpha)
                lam = gamma1 / (gamma1 + gamma2)
                batch_size = tf.shape(X)[0]
                index = tf.random.shuffle(tf.range(batch_size))
                X_mix = lam * X + (1.0 - lam) * tf.gather(X, index)
                y_mix = lam * y_true + (1.0 - lam) * tf.gather(y_true, index)
            else:
                X_mix, y_mix = X, y_true
        else:
            X_mix, y_mix = X, y_true

        with tf.GradientTape() as tape:
            preds = model(X_mix, training=True)
            loss = loss_fn(y_mix, preds)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.reduce_mean(loss)
    return train_step

def train_fold(fold):
    print(f"\n{'='*50}\n训练第 {fold+1}/{config.N_FOLDS} 折\n{'='*50}")
    train_dataset = ESC50Dataset(fold=fold, mode='train', augment=True)
    val_dataset = ESC50Dataset(fold=fold, mode='val', augment=False)
    train_ds = train_dataset.dataset
    val_ds = val_dataset.dataset
    train_samples = len(train_dataset.target_segments) + int(len(train_dataset.target_segments) * config.OTHER_RATIO)
    val_samples = len(val_dataset.target_segments) + int(len(val_dataset.target_segments) * config.OTHER_RATIO)
    steps_per_epoch = max(1, train_samples // config.BATCH_SIZE)
    validation_steps = max(1, val_samples // config.BATCH_SIZE)
    print(f"训练样本约: {train_samples}, 验证样本约: {val_samples}")

    with strategy.scope():
        model = create_tiny_model(alpha=0.5, dropout=0.3)
        base_optimizer = tf.keras.optimizers.Adam(
            learning_rate=CosineAnnealingScheduler(
                config.MAX_LEARNING_RATE, config.MIN_LEARNING_RATE,
                config.WARMUP_EPOCHS, config.EPOCHS, steps_per_epoch
            )
        )
        if config.MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            use_mp = True
        else:
            optimizer = base_optimizer
            use_mp = False

    train_step_fn = create_train_step(model, optimizer, use_mixed_precision=use_mp)
    best_val_acc = 0.0
    fold_h5 = f"models/saved/fold_{fold}_best.h5"

    for epoch in range(config.EPOCHS):
        start = time.time()
        train_loss = 0.0
        batches = 0
        for step, (X_batch, y_batch) in enumerate(train_ds.take(steps_per_epoch)):
            loss = train_step_fn(X_batch, y_batch)
            train_loss += float(loss)
            batches += 1
            if step % 20 == 0:
                print(f"  Epoch {epoch+1}/{config.EPOCHS} | Step {step}/{steps_per_epoch} | Loss: {float(loss):.4f}")
        avg_train_loss = train_loss / max(batches, 1)

        val_correct = 0
        val_total = 0
        for X_val, y_val in val_ds.take(validation_steps):
            probs = model(X_val, training=False)
            y_pred = tf.argmax(probs, axis=1)
            y_true = tf.argmax(y_val, axis=1)
            val_correct += tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.int32))
            val_total += tf.shape(y_val)[0]
        val_acc = float(val_correct) / float(val_total)
        epoch_time = time.time() - start
        print(f"  Epoch {epoch+1} 完成 | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(fold_h5)
            print(f"  -> 保存最佳模型 (acc={val_acc:.4f})")

    # 最终评估
    model.load_weights(fold_h5)
    X_val_list, y_val_list = [], []
    for X_batch, y_batch in val_ds:
        X_val_list.append(X_batch.numpy())
        y_val_list.append(y_batch.numpy())
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)
    y_pred_all = np.argmax(model.predict(X_val_all, verbose=0, batch_size=config.BATCH_SIZE), axis=1)
    y_true_all = np.argmax(y_val_all, axis=1)

    acc = accuracy_score(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)
    f1_weighted = report['weighted avg']['f1-score']
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"  折 {fold} 最终验证准确率: {acc:.4f}  F1(weighted): {f1_weighted:.4f}")
    print("  混淆矩阵:\n", cm)
    print_gpu_memory_usage()
    return best_val_acc, f1_weighted, model

def main():
    print("=" * 60)
    print("多分类音频事件检测模型训练（含 other 类，在线增强 + 自定义背景音）")
    print("=" * 60)

    fold_accs = []
    fold_f1s = []
    for fold in range(config.N_FOLDS):
        best_val_acc, f1, model = train_fold(fold)
        fold_accs.append(best_val_acc)
        fold_f1s.append(f1)

    print(f"\n{'='*60}\n交叉验证完成")
    print(f"各折最佳准确率: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"平均最佳准确率: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    print(f"平均F1(weighted): {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")

    # 选择最佳折（按F1）
    best_fold = np.argmax(fold_f1s)
    best_h5 = f"models/saved/fold_{best_fold}_best.h5"
    print(f"选择折 {best_fold} 作为最佳模型")
    best_model = tf.keras.models.load_model(best_h5)
    best_model.save(config.MODEL_SAVE_PATH)
    print(f"模型已保存: {config.MODEL_SAVE_PATH}")

    # TFLite 导出
    print("转换为 TFLite 格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(config.TFLITE_MODEL_PATH) / 1024
    print(f"TFLite 模型已保存: {config.TFLITE_MODEL_PATH}，大小: {size_kb:.1f} KB")
    print_gpu_memory_usage()

if __name__ == "__main__":
    main()
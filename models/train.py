"""
针对 ESC-50 小数据集的训练脚本：五折交叉验证 + 在线数据增强。
最终保存最佳模型为 .h5 和 .tflite，并输出平均准确率。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score
import config
from features.mel_spectrogram import MelSpectrogram
# 根据特征提取器参数动态修正模型输入形状
_mel_ext = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, _mel_ext.eff_n_mels, 1)

from data.dataset import ESC50Dataset
from models.lightweight_model import create_tiny_model

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)


def train_fold(fold):
    """训练单个折，返回最佳 val_loss 和对应的模型。"""
    train_gen = ESC50Dataset(fold=fold, mode='train', augment=True)
    val_gen = ESC50Dataset(fold=fold, mode='val', augment=False)

    # 计算每折的 steps
    train_samples = len(train_gen.get_fold_data(fold))
    val_samples = len(val_gen.get_fold_data(fold))
    steps_per_epoch = max(1, train_samples // config.BATCH_SIZE)
    validation_steps = max(1, val_samples // config.BATCH_SIZE)

    model = create_tiny_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 保存每折最佳模型
    fold_h5 = f"models/saved/fold_{fold}.h5"
    callbacks = [
        ModelCheckpoint(fold_h5, monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen.generator(config.BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=val_gen.generator(config.BATCH_SIZE),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # 加载最佳权重进行评估
    best_val_loss = min(history.history['val_loss'])
    fold_val_acc = max(history.history['val_accuracy'])

    # 额外全验证集准确率
    model.load_weights(fold_h5)
    all_val_data = val_gen.get_fold_data(fold)
    X_val, y_val = _extract_all(all_val_data, val_gen.mel_extractor)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_true, y_pred)

    return best_val_loss, acc, model, fold_h5


def _extract_all(samples, mel_extractor):
    """从样本列表批量提取固定形状的特征。"""
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
    fold_losses = []
    fold_accs = []
    best_model_path = None

    for fold in range(config.N_FOLDS):
        print(f"\n===== 训练第 {fold+1}/{config.N_FOLDS} 折 =====")
        val_loss, acc, model, h5_path = train_fold(fold)
        fold_losses.append(val_loss)
        fold_accs.append(acc)
        print(f"折 {fold} 验证准确率: {acc:.4f}")
        # 保存最后一折的 H5 作为最终模型（或选择最佳折）
        if best_model_path is None or val_loss < min(fold_losses):
            best_model_path = h5_path
            model.save(config.MODEL_SAVE_PATH)  # 覆盖保存最佳 H5

    print(f"\n===== 交叉验证完成 =====")
    print(f"平均准确率: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # 转换为 TFLite
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # 这里先不做量化，量化模块后续扩展
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(config.TFLITE_MODEL_PATH), exist_ok=True)
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite 模型已保存至 {config.TFLITE_MODEL_PATH}")
    print(f"模型大小: {os.path.getsize(config.TFLITE_MODEL_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
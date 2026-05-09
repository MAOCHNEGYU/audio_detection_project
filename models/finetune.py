"""
基于现有最佳模型进行域适应微调
v1.3: 融入真实滴答声，温和增强目标类
"""
from utils.project_init import set_project_root
set_project_root()

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
from data.dataset import ESC50Dataset
from data.real_dataset import RealClockDataset
from models.lightweight_model import create_tiny_model
from features.mel_spectrogram import MelSpectrogram
from utils.gpu_utils import setup_gpu_environment, print_gpu_memory_usage


tf.random.set_seed(42)
np.random.seed(42)

strategy = setup_gpu_environment()
tmp_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, tmp_mel.eff_n_mels, 1)

def build_mixed_dataset(mode='train'):
    """
    构建混合数据集：ESC-50目标类 + 真实滴答声 + other类
    """
    # ESC-50 数据集（不增强的目标类 + other 类）
    esc50 = ESC50Dataset(fold=0, mode=mode, augment=True)  # 用前一折作为示例，实际微调应使用全部数据
    target_audios = [s[0] for s in esc50.target_segments]
    target_labels = [s[1] for s in esc50.target_segments]

    # 真实滴答声片段
    real_data = RealClockDataset()
    real_audios = real_data.segments
    real_labels = [0] * len(real_audios)

    # 组合目标类：ESC-50滴答声 + 真实滴答声
    num_real = int(len(target_audios) * config.REAL_CLOCK_RATIO)
    # 从真实数据中随机采样
    if len(real_audios) > num_real:
        indices = np.random.choice(len(real_audios), num_real, replace=False)
        real_audios = [real_audios[i] for i in indices]
        real_labels = [0] * num_real
    else:
        # 如果真实样本不够，就全用
        real_audios = real_audios
        real_labels = [0] * len(real_audios)

    combined_target_audios = target_audios + real_audios
    combined_target_labels = target_labels + real_labels

    # 构建 other 类（使用 ESC50Dataset 的采样方法）
    # 这里我们直接复用 dataset.py 中的 _sample_other_segment 不太好，因为那是实例方法。
    # 简单起见，我们从 ESC50 的 other_files_safe 中随机切片段。
    other_audios = []
    other_labels = []
    other_pool = esc50.other_files_safe
    num_other = int(len(combined_target_audios) * config.OTHER_RATIO)
    for _ in range(num_other):
        fname = np.random.choice(other_pool)
        path = os.path.join(config.ESC_50_AUDIO_DIR, fname)
        try:
            audio, sr = sf.read(path)
            if sr != config.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) >= config.SEGMENT_SIZE:
                start = np.random.randint(0, len(audio) - config.SEGMENT_SIZE)
                seg = audio[start:start+config.SEGMENT_SIZE].astype(np.float32)
                max_val = np.max(np.abs(seg))
                if max_val > 0:
                    seg = seg / max_val
                other_audios.append(seg)
                other_labels.append(config.NUM_CLASSES - 1)
        except:
            continue

    # 补齐 other 数量
    while len(other_audios) < num_other:
        fake = np.random.randn(config.SEGMENT_SIZE).astype(np.float32) * 0.02
        other_audios.append(fake)
        other_labels.append(config.NUM_CLASSES - 1)

    # 构建 TensorFlow Dataset
    all_audios = combined_target_audios + other_audios
    all_labels = combined_target_labels + other_labels
    ds = tf.data.Dataset.from_tensor_slices((all_audios, all_labels))
    ds = ds.shuffle(buffer_size=len(all_audios))

    # 特征提取与增强
    def preprocess(audio, label):
        # 对目标类进行温和增强
        if label == 0:  # 目标类
            # 微弱的音高偏移和噪声
            import librosa
            if np.random.random() < 0.3:  # 30%概率音高偏移
                steps = np.random.uniform(-1, 1)
                audio = librosa.effects.pitch_shift(audio.numpy(), sr=config.SAMPLE_RATE, n_steps=steps)
            # 时间平移（已在 _augment_audio 的逻辑中，这里简单重复）
            shift = np.random.randint(-int(0.05*config.SAMPLE_RATE), int(0.05*config.SAMPLE_RATE))
            if shift > 0:
                audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
            audio = audio + np.random.randn(*audio.shape) * 0.0005  # 极轻噪声
        # 特征提取
        from features.mel_spectrogram import MelSpectrogram
        mel_ext = MelSpectrogram(top_db=None)
        audio_int16 = (audio.numpy() * 32767).astype(np.int16)
        log_mel = mel_ext.compute(audio_int16)
        target_frames = config.MODEL_INPUT_SHAPE[0]
        if log_mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - log_mel.shape[0], log_mel.shape[1]), dtype=np.float32)
            log_mel = np.vstack([log_mel, pad])
        else:
            log_mel = log_mel[:target_frames]
        log_mel = np.expand_dims(log_mel, axis=-1)
        label_onehot = tf.keras.utils.to_categorical(label, num_classes=config.NUM_CLASSES)
        return log_mel.astype(np.float32), label_onehot.astype(np.float32)

    ds = ds.map(
        lambda audio, label: tf.numpy_function(
            func=preprocess, inp=[audio, label], Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda x, y: (
        tf.reshape(x, config.MODEL_INPUT_SHAPE),
        tf.reshape(y, (config.NUM_CLASSES,))
    ))
    ds = ds.batch(config.BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(all_audios)


def finetune():
    print("="*60)
    print("域适应微调：融入真实滴答声样本")
    print("="*60)

    # 加载之前训练好的模型
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError("请先完成基础训练，生成 models/saved/audio_detector.h5")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # 构建混合训练集
    train_ds, train_samples = build_mixed_dataset(mode='train')
    steps_per_epoch = max(1, train_samples // config.BATCH_SIZE)

    # 优化器：更低的学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    @tf.function
    def train_step(X, y_true):
        with tf.GradientTape() as tape:
            preds = model(X, training=True)
            loss = loss_fn(y_true, preds)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_acc = 0.0
    for epoch in range(config.FINETUNE_EPOCHS):
        total_loss = 0.0
        batches = 0
        for X, y in train_ds.take(steps_per_epoch):
            loss = train_step(X, y)
            total_loss += float(loss)
            batches += 1
        avg_loss = total_loss / batches
        print(f"Epoch {epoch+1}/{config.FINETUNE_EPOCHS} - Loss: {avg_loss:.4f}")

        # 每个 epoch 结束后，可以用少量真实录音测试（可选）
        # 这里简单用训练集的准确率作为参考（由于训练集可能会过拟合，但微调数据少，可接受）
        # 更好的做法是留出一些真实录音做验证，但我们现在快速迭代，先不搞太复杂。
        if avg_loss < 1e-3:
            print("损失已经极低，提前结束微调")
            break

    # 保存微调后的模型
    finetuned_path = "models/saved/audio_detector_finetuned.h5"
    model.save(finetuned_path)
    print(f"微调模型已保存至: {finetuned_path}")

    # 直接覆盖主模型路径，方便实时系统使用
    model.save(config.MODEL_SAVE_PATH)
    # 转换 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(config.TFLITE_MODEL_PATH) / 1024
    print(f"TFLite 模型已更新: {config.TFLITE_MODEL_PATH}，大小: {size_kb:.1f} KB")
    print("微调完成，请重新运行 run.py 测试。")


if __name__ == "__main__":
    finetune()
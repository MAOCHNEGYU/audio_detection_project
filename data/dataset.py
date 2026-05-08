"""
ESC-50 小数据集加载与在线增强模块。
每一条原始音频（5秒）将被切割为多个 1 秒重叠片段，生成更多样本。
同时应用时间拉伸、音高偏移、背景噪声混合、SpecAugment 等增强策略。
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import config
from features.mel_spectrogram import MelSpectrogram
import tensorflow as tf

class ESC50Dataset:
    """
    加载 ESC-50 中指定四类，生成 (log-mel, label) 样本的生成器。
    支持交叉验证折的划分。
    """
    def __init__(self, fold=None, mode='train', augment=True):
        """
        Args:
            fold: int 或 None。指定使用哪一折作为验证集（0-4），None 则返回全部。
            mode: 'train' 或 'val'。训练集应用增强，验证集不增强。
            augment: 是否应用数据增强（训练模式下通常为 True）。
        """
        self.fold = fold
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.mel_extractor = MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
            low_freq_cut=config.LOW_FREQ_CUT,
            top_db=None  # 训练时不裁剪，保留信息
        )
        self.meta = self._load_meta()
        self.file_list, self.labels = self._filter_targets()
        self.samples = self._split_segments()  # 生成 (音频段, 标签) 列表

    def _load_meta(self):
        """读取 ESC-50 元数据并处理类别映射。"""
        df = pd.read_csv(config.ESC_50_META_PATH)
        # 映射目标类别到数字 id
        self.class_map = {name: i for i, name in enumerate(config.TARGET_CLASSES.keys())}
        self.reverse_map = {v: k for k, v in self.class_map.items()}
        return df

    def _filter_targets(self):
        """筛选出四类音频，并转换为内部标签。"""
        # 构建 esc50 类别名 -> 我们的类别名的映射
        target_esc50_names = []
        for our_name, esc_names in config.TARGET_CLASSES.items():
            target_esc50_names.extend(esc_names)
        mask = self.meta['category'].isin(target_esc50_names)
        filtered = self.meta[mask].copy()
        # 生成数字标签
        labels = []
        for cat in filtered['category']:
            for our_name, esc_names in config.TARGET_CLASSES.items():
                if cat in esc_names:
                    labels.append(self.class_map[our_name])
                    break
        filtered['label'] = labels
        return filtered['filename'].values, filtered['label'].values

    def _split_segments(self):
        """
        将每条 5 秒音频切分为多个 1 秒重叠片段（步长 0.5 秒），
        并按五折交叉验证分配到训练/验证集。
        """
        segments = []  # (file, start_sample, label)
        for fname, label in zip(self.file_list, self.labels):
            path = os.path.join(config.ESC_50_AUDIO_DIR, fname)
            audio, sr = sf.read(path)
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            # 若为立体声，转单声道
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # 切分：从0开始，步长 0.5 秒
            segment_len = config.SEGMENT_SIZE       # 16000
            step = segment_len // 2                 # 8000，0.5秒步长
            for start in range(0, len(audio) - segment_len + 1, step):
                segments.append((audio[start:start+segment_len], label))
        np.random.shuffle(segments)  # 打乱顺序
        return segments

    def get_fold_data(self, fold):
        """
        返回 5 折中的训练/验证样本列表。
        fold: 当前作为验证集的折编号 (0-4)
        """
        n = len(self.samples)
        indices = np.arange(n)
        fold_size = n // config.N_FOLDS
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < config.N_FOLDS - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.setdiff1d(indices, val_idx)
        if self.mode == 'train':
            return [self.samples[i] for i in train_idx]
        else:
            return [self.samples[i] for i in val_idx]

    def _augment_sample(self, audio):
        """
        对单段音频应用随机增强（时间拉伸、音高偏移、加噪、SpecAugment）。
        """
        # 时间拉伸
        if np.random.random() < 0.5:
            rate = np.random.uniform(*config.AUG_TIME_STRETCH)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            if len(audio) < config.SEGMENT_SIZE:
                audio = np.pad(audio, (0, config.SEGMENT_SIZE - len(audio)), mode='constant')
            else:
                audio = audio[:config.SEGMENT_SIZE]

        # 音高偏移
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(*config.AUG_PITCH_SHIFT)
            audio = librosa.effects.pitch_shift(audio, sr=config.SAMPLE_RATE, n_steps=n_steps)

        # 混合随机背景噪声（从 ESC-50 其他类随机选取）
        if np.random.random() < 0.7:
            noise_file = np.random.choice(self.file_list)
            noise_path = os.path.join(config.ESC_50_AUDIO_DIR, noise_file)
            noise, sr = sf.read(noise_path)
            if sr != config.SAMPLE_RATE:
                noise = librosa.resample(noise, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            # 随机截取
            if len(noise) > config.SEGMENT_SIZE:
                start = np.random.randint(0, len(noise) - config.SEGMENT_SIZE)
                noise = noise[start:start+config.SEGMENT_SIZE]
            else:
                noise = np.pad(noise, (0, config.SEGMENT_SIZE - len(noise)), mode='constant')
            snr = np.random.uniform(*config.AUG_NOISE_SNR)
            noise_power = np.mean(noise**2)
            sig_power = np.mean(audio**2)
            scale = np.sqrt(sig_power / (noise_power * 10**(snr/10) + 1e-10))
            audio = audio + scale * noise
            audio = np.clip(audio, -1.0, 1.0)

        return audio.astype(np.float32)

    def _apply_specaugment(self, log_mel):
        """在 log-mel 谱图上进行时间/频率掩蔽。"""
        # 频域掩蔽
        if np.random.random() < 0.5:
            f_max = min(log_mel.shape[1] - 1, 8)
            f_start = np.random.randint(0, log_mel.shape[1] - f_max)
            log_mel[:, f_start:f_start + np.random.randint(1, f_max)] = 0
        # 时域掩蔽
        if np.random.random() < 0.5:
            t_max = min(log_mel.shape[0] - 1, 10)
            t_start = np.random.randint(0, log_mel.shape[0] - t_max)
            log_mel[t_start:t_start + np.random.randint(1, t_max), :] = 0
        return log_mel

    def generator(self, batch_size=32):
        """生成器，每 epoch 无限迭代。"""
        fold = self.fold
        if fold is not None:
            samples = self.get_fold_data(fold)
        else:
            samples = self.samples.copy()
            np.random.shuffle(samples)

        while True:
            np.random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]
                X_batch = []
                Y_batch = []
                for audio, label in batch_samples:
                    # 训练时应用增强
                    if self.augment:
                        audio = self._augment_sample(audio)
                    # 提取 log-mel
                    mel = self.mel_extractor.compute((audio * 32768).astype(np.int16))
                    # 固定帧数
                    target_frames = config.MODEL_INPUT_SHAPE[0]
                    if mel.shape[0] < target_frames:
                        pad = np.zeros((target_frames - mel.shape[0], mel.shape[1]), dtype=np.float32)
                        mel = np.vstack([mel, pad])
                    else:
                        mel = mel[:target_frames]
                    if self.augment and config.AUG_SPEC_AUGMENT:
                        mel = self._apply_specaugment(mel)
                    X_batch.append(mel)
                    Y_batch.append(label)
                X_batch = np.array(X_batch, dtype=np.float32)[..., np.newaxis]  # 加通道
                Y_batch = tf.keras.utils.to_categorical(Y_batch, num_classes=config.NUM_CLASSES)
                yield X_batch, Y_batch
"""
ESC-50 小数据集加载与在线增强模块。
改进：使用 tf.data.Dataset 替代 Python 生成器，支持 GPU 预取和并行预处理。
修复：get_fold_data 不再依赖 self.samples 自身，避免递归混乱。
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
import config
from features.mel_spectrogram import MelSpectrogram


class ESC50Dataset:
    """
    加载 ESC-50 中指定类别，生成 (log-mel, label) 样本。
    改进版：内部使用 tf.data.Dataset 实现高性能数据管道，支持 GPU 预取。
    """

    def __init__(self, fold=None, mode='train', augment=True):
        """
        Args:
            fold: int 或 None。指定验证折（0-4），None 则返回全部。
            mode: 'train' 或 'val'。训练集应用增强，验证集不增强。
            augment: 是否应用数据增强（训练模式下通常为 True）。
        """
        self.fold = fold
        self.mode = mode
        self.augment = augment and (mode == 'train')

        # 初始化 Mel 提取器（用于离线预处理缓存）
        self.mel_extractor = MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
            low_freq_cut=config.LOW_FREQ_CUT,
            top_db=None          # 训练时不裁剪，保留信息
        )

        # 加载元数据并筛选目标类别
        self.meta = self._load_meta()
        self.file_list, self.labels = self._filter_targets()

        # 保存全部切分后的样本（未按折划分）
        self.all_samples = self._split_segments()

        # 根据 fold 划分当前要使用的样本
        if fold is not None:
            self.samples = self._get_fold_split(fold, self.all_samples)
        else:
            self.samples = self.all_samples.copy()

        # 构建 tf.data.Dataset 管道
        self.dataset = self._build_tf_dataset()

    def _load_meta(self):
        """读取 ESC-50 元数据并处理类别映射。"""
        df = pd.read_csv(config.ESC_50_META_PATH)
        self.class_map = {name: i for i, name in enumerate(config.TARGET_CLASSES.keys())}
        self.reverse_map = {v: k for k, v in self.class_map.items()}
        return df

    def _filter_targets(self):
        """筛选目标类别音频并转换为数字标签。"""
        target_esc50_names = []
        for our_name, esc_names in config.TARGET_CLASSES.items():
            target_esc50_names.extend(esc_names)

        mask = self.meta['category'].isin(target_esc50_names)
        filtered = self.meta[mask].copy()

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
        将每条 5 秒音频切分为 1 秒重叠片段（步长 0.5 秒）。
        返回: list of (audio_segment, label)，audio_segment 为 float32 数组
        """
        segments = []
        segment_len = config.SEGMENT_SIZE    # 16000
        step = segment_len // 2              # 8000，0.5 秒步长

        for fname, label in zip(self.file_list, self.labels):
            path = os.path.join(config.ESC_50_AUDIO_DIR, fname)

            try:
                audio, sr = sf.read(path)
            except Exception as e:
                print(f"读取文件失败 {path}: {e}")
                continue

            # 重采样到 16kHz
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

            # 转单声道
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # 切分 1 秒片段
            for start in range(0, len(audio) - segment_len + 1, step):
                segment = audio[start:start + segment_len].astype(np.float32)
                # 归一化到 [-1, 1]（与 mel_extractor 内部一致）
                if np.max(np.abs(segment)) > 0:
                    segment = segment / np.max(np.abs(segment))
                segments.append((segment, label))

        np.random.shuffle(segments)
        return segments

    def _get_fold_split(self, fold, full_samples):
        """
        根据折编号从全样本列表中划分出训练/验证子集。
        此方法不修改 self.samples，仅返回划分后的列表。
        """
        n = len(full_samples)
        indices = np.arange(n)
        fold_size = n // config.N_FOLDS

        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < config.N_FOLDS - 1 else n

        if self.mode == 'train':
            train_idx = np.setdiff1d(indices, indices[val_start:val_end])
            return [full_samples[i] for i in train_idx]
        else:
            return [full_samples[i] for i in range(val_start, val_end)]

    def get_fold_data(self, fold):
        """返回指定折的数据列表（用于验证集全量评估）。"""
        return self._get_fold_split(fold, self.all_samples)

    # ---------- 以下为数据增强与预处理 ----------
    def _augment_sample(self, audio):
        """对音频应用随机增强（时间拉伸、音高偏移、加噪）。"""
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
            audio = librosa.effects.pitch_shift(
                audio, sr=config.SAMPLE_RATE, n_steps=n_steps
            )

        # 混合背景噪声
        if np.random.random() < 0.7:
            noise_file = np.random.choice(self.file_list)
            noise_path = os.path.join(config.ESC_50_AUDIO_DIR, noise_file)
            try:
                noise, sr = sf.read(noise_path)
                if sr != config.SAMPLE_RATE:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                if noise.ndim > 1:
                    noise = noise.mean(axis=1)

                if len(noise) > config.SEGMENT_SIZE:
                    start = np.random.randint(0, len(noise) - config.SEGMENT_SIZE)
                    noise = noise[start:start + config.SEGMENT_SIZE]
                else:
                    noise = np.pad(noise, (0, config.SEGMENT_SIZE - len(noise)), mode='constant')

                snr = np.random.uniform(*config.AUG_NOISE_SNR)
                noise_power = np.mean(noise ** 2) + 1e-10
                sig_power = np.mean(audio ** 2) + 1e-10
                scale = np.sqrt(sig_power / (noise_power * (10 ** (snr / 10))))
                audio = audio + scale * noise
                audio = np.clip(audio, -1.0, 1.0)
            except Exception:
                pass

        return audio.astype(np.float32)

    def _extract_mel(self, audio):
        """提取 Mel 特征并固定帧数。"""
        audio_int16 = (audio * 32767).astype(np.int16)
        mel = self.mel_extractor.compute(audio_int16)

        target_frames = config.MODEL_INPUT_SHAPE[0]    # 49

        if mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - mel.shape[0], mel.shape[1]), dtype=np.float32)
            mel = np.vstack([mel, pad])
        else:
            mel = mel[:target_frames]

        return mel.astype(np.float32)

    def _apply_specaugment(self, log_mel):
        """在 log-mel 谱图上进行时间/频率掩蔽。"""
        # 频域掩蔽
        if np.random.random() < 0.5:
            f_max = min(log_mel.shape[1] - 1, 8)
            if f_max > 0:
                f_start = np.random.randint(0, log_mel.shape[1] - f_max)
                width = np.random.randint(1, f_max + 1)
                log_mel[:, f_start:f_start + width] = 0

        # 时域掩蔽
        if np.random.random() < 0.5:
            t_max = min(log_mel.shape[0] - 1, 10)
            if t_max > 0:
                t_start = np.random.randint(0, log_mel.shape[0] - t_max)
                width = np.random.randint(1, t_max + 1)
                log_mel[t_start:t_start + width, :] = 0

        return log_mel

    def _preprocess_sample(self, audio, label):
        """
        单样本预处理函数（用于 tf.data map）。
        包含增强、特征提取、SpecAugment。
        """
        if self.augment:
            audio = self._augment_sample(audio)

        mel = self._extract_mel(audio)

        if self.augment and config.AUG_SPEC_AUGMENT:
            mel = self._apply_specaugment(mel)

        mel = np.expand_dims(mel, axis=-1)              # (49, 32) -> (49, 32, 1)
        label_onehot = tf.keras.utils.to_categorical(label, num_classes=config.NUM_CLASSES)

        return mel.astype(np.float32), label_onehot.astype(np.float32)

    def _build_tf_dataset(self):
        """构建 tf.data.Dataset 高性能数据管道。"""
        audios = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        ds = tf.data.Dataset.from_tensor_slices((audios, labels))

        ds = ds.map(
            lambda audio, label: tf.numpy_function(
                func=self._preprocess_sample,
                inp=[audio, label],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 明确输出形状
        ds = ds.map(lambda x, y: (
            tf.reshape(x, config.MODEL_INPUT_SHAPE),
            tf.reshape(y, (config.NUM_CLASSES,))
        ))

        if self.mode == 'train':
            ds = ds.shuffle(buffer_size=min(1000, len(self.samples)))

        ds = ds.batch(config.BATCH_SIZE, drop_remainder=True)

        if config.CACHE_DATASET and len(self.samples) < 10000:
            ds = ds.cache()

        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

    def generator(self, batch_size=32):
        """兼容旧接口：返回 tf.data.Dataset（本身就是可迭代对象）。"""
        return self.dataset

    # 注意：get_fold_data 已置于类的前部，供 train.py 中的 _extract_all 调用
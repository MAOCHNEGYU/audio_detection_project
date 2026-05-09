"""
ESC-50 在线增强数据集，动态构建 other 类，并支持自定义背景音。
v1.1: 关闭破坏性增强，other类以纯净环境音/弱噪声为主
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
    def __init__(self, fold=None, mode='train', augment=True):
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
            top_db=None
        )
        meta = pd.read_csv(config.ESC_50_META_PATH)
        self.class_map = {name: i for i, name in enumerate(config.TARGET_CLASSES.keys())}
        self.other_idx = config.NUM_CLASSES - 1

        all_files = meta['filename'].values
        all_categories = meta['category'].values

        target_names = []
        for names in config.TARGET_CLASSES.values():
            target_names.extend(names)

        self.target_files = []
        self.target_labels = []
        self.other_files = []
        for f, cat in zip(all_files, all_categories):
            if cat in target_names:
                self.target_files.append(f)
                for cls, names in config.TARGET_CLASSES.items():
                    if cat in names:
                        self.target_labels.append(self.class_map[cls])
            else:
                self.other_files.append(f)

        # 切分目标样本
        self.target_segments = self._split_segments(self.target_files, self.target_labels)

        # 加载自定义背景音（优先用作 other 类）
        self.background_pool = []
        if hasattr(config, 'BACKGROUND_FILES'):
            for bg_file in config.BACKGROUND_FILES:
                if os.path.exists(bg_file):
                    try:
                        audio, sr = sf.read(bg_file)
                        if sr != config.SAMPLE_RATE:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                        if audio.ndim > 1:
                            audio = audio.mean(axis=1)
                        if len(audio) >= config.SEGMENT_SIZE:
                            self.background_pool.append(audio.astype(np.float32))
                    except Exception as e:
                        print(f"背景音加载失败 {bg_file}: {e}")

        # 按折划分目标样本
        if fold is not None:
            self.target_segments = self._get_fold_split(fold, self.target_segments)

        self.dataset = self._build_dataset()

    def _split_segments(self, file_list, labels):
        segments = []
        segment_len = config.SEGMENT_SIZE
        step = segment_len // 2
        for fname, label in zip(file_list, labels):
            path = os.path.join(config.ESC_50_AUDIO_DIR, fname)
            try:
                audio, sr = sf.read(path)
            except:
                continue
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            for start in range(0, len(audio) - segment_len + 1, step):
                seg = audio[start:start+segment_len].astype(np.float32)
                max_val = np.max(np.abs(seg))
                if max_val > 0:
                    seg = seg / max_val
                segments.append((seg, label))
        np.random.shuffle(segments)
        return segments

    def _get_fold_split(self, fold, full_samples):
        n = len(full_samples)
        indices = np.arange(n)
        fold_size = n // config.N_FOLDS
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < config.N_FOLDS - 1 else n
        if self.mode == 'train':
            idx = np.setdiff1d(indices, indices[val_start:val_end])
        else:
            idx = indices[val_start:val_end]
        return [full_samples[i] for i in idx]

    def _augment_audio(self, audio):
        # v1.1: 只保留极微弱的噪声注入，不做其他任何增强
        # 加入低强度白噪声，信噪比 30~40 dB，仅用于增加一点鲁棒性
        snr = np.random.uniform(*config.AUG_NOISE_SNR)
        noise = np.random.randn(len(audio)).astype(np.float32)
        sig_power = np.mean(audio ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(sig_power / (noise_power * (10 ** (snr / 10))))
        audio = audio + scale * noise
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)

    def _extract_mel(self, audio):
        audio_int16 = (audio * 32767).astype(np.int16)
        mel = self.mel_extractor.compute(audio_int16)
        target_frames = config.MODEL_INPUT_SHAPE[0]
        if mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - mel.shape[0], mel.shape[1]), dtype=np.float32)
            mel = np.vstack([mel, pad])
        else:
            mel = mel[:target_frames]
        return mel.astype(np.float32)

    def _apply_specaugment(self, log_mel):
        # 暂时完全禁用
        return log_mel

    def _preprocess(self, audio, label):
        if self.augment:
            audio = self._augment_audio(audio)
        mel = self._extract_mel(audio)
        mel = self._apply_specaugment(mel)
        mel = np.expand_dims(mel, axis=-1)
        label_onehot = tf.keras.utils.to_categorical(label, num_classes=config.NUM_CLASSES)
        return mel.astype(np.float32), label_onehot.astype(np.float32)

    def _sample_other_segment(self):
        # 优先使用自定义背景音，若无则生成极弱噪声（防止构造假滴答声）
        if self.background_pool and np.random.random() < 0.8:
            full_audio = self.background_pool[np.random.randint(len(self.background_pool))]
            if len(full_audio) > config.SEGMENT_SIZE:
                start = np.random.randint(0, len(full_audio) - config.SEGMENT_SIZE)
                seg = full_audio[start:start + config.SEGMENT_SIZE]
            else:
                seg = full_audio.copy()
            # 添加微量噪声避免过静
            seg = seg + 0.001 * np.random.randn(len(seg)).astype(np.float32)
            return seg, self.other_idx
        else:
            # 纯高斯噪声作为 other 类（幅度小）
            fake = np.random.randn(config.SEGMENT_SIZE).astype(np.float32) * 0.01
            return fake, self.other_idx

    def _build_dataset(self):
        target_audios = [s[0] for s in self.target_segments]
        target_labels = [s[1] for s in self.target_segments]
        target_ds = tf.data.Dataset.from_tensor_slices((target_audios, target_labels))

        num_other = int(len(self.target_segments) * config.OTHER_RATIO)
        other_audios = []
        other_labels = []
        for _ in range(num_other):
            aud, lab = self._sample_other_segment()
            other_audios.append(aud)
            other_labels.append(lab)
        other_ds = tf.data.Dataset.from_tensor_slices((other_audios, other_labels))

        ds = target_ds.concatenate(other_ds)
        if self.mode == 'train':
            ds = ds.shuffle(buffer_size=min(1000, len(target_audios)+num_other))

        ds = ds.map(
            lambda audio, label: tf.numpy_function(
                func=self._preprocess,
                inp=[audio, label],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(lambda x, y: (
            tf.reshape(x, config.MODEL_INPUT_SHAPE),
            tf.reshape(y, (config.NUM_CLASSES,))
        ))
        ds = ds.batch(config.BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def generator(self, batch_size=32):
        return self.dataset
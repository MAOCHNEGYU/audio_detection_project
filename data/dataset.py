"""
ESC-50 在线增强数据集，动态构建 other 类，并支持自定义背景音。
v1.2: other类注入真实但非脉冲音频，目标类只做极温和增强
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

        # 收集所有容易与“滴答声”混淆的类别，将它们从 other 池中剔除
        confusable_categories = [
            'keyboard_clicks', 'metronome', 'typing', 'mouse_clicks',
            'clapping', 'church_bells', 'clock_alarm', 'door_wood_knocks',
            'footsteps', 'knock', 'squeak', 'tearing'
        ]

        self.target_files = []
        self.target_labels = []
        self.other_files_full = []  # 所有非目标类文件
        self.other_files_safe = []  # 剔除易混淆类后的安全文件
        for f, cat in zip(all_files, all_categories):
            if cat in target_names:
                self.target_files.append(f)
                for cls, names in config.TARGET_CLASSES.items():
                    if cat in names:
                        self.target_labels.append(self.class_map[cls])
            else:
                self.other_files_full.append(f)
                if cat not in confusable_categories:
                    self.other_files_safe.append(f)

        # 如果剔除后其他类文件太少，回退到使用全部文件
        if len(self.other_files_safe) < 50:
            print("警告：剔除易混淆类后剩余样本过少，使用全部非目标文件")
            self.other_files_safe = self.other_files_full

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
                seg = audio[start:start + segment_len].astype(np.float32)
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

    def _augment_audio(self, audio, is_target=True):
        """
        v1.2: 目标类只做极温和增强，other类不做任何增强
        - 目标类：极轻噪声 + 时间平移（保持节奏规律）
        - other类：原样返回（其多样性来自数据源本身）
        """
        if not is_target:
            return audio.astype(np.float32)

        # 1. 极轻的高斯噪声 (SNR 35~45 dB, 几乎听不见)
        snr = np.random.uniform(35, 45)
        noise = np.random.randn(len(audio)).astype(np.float32)
        sig_power = np.mean(audio ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(sig_power / (noise_power * (10 ** (snr / 10))))
        audio = audio + scale * noise

        # 2. 时间平移（±0.05秒），而不是时间拉伸
        # 这保持了滴答声的节奏规律，只改变了起始点
        max_shift_samples = int(0.05 * config.SAMPLE_RATE)  # 0.05秒
        shift = np.random.randint(-max_shift_samples, max_shift_samples)
        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
        elif shift < 0:
            audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]

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
        # 判断是否为目标类
        is_target = (label != self.other_idx)
        if self.augment:
            audio = self._augment_audio(audio, is_target=is_target)
        mel = self._extract_mel(audio)
        mel = self._apply_specaugment(mel)
        mel = np.expand_dims(mel, axis=-1)
        label_onehot = tf.keras.utils.to_categorical(label, num_classes=config.NUM_CLASSES)
        return mel.astype(np.float32), label_onehot.astype(np.float32)

    def _sample_other_segment(self):
        """
        v1.2: 混合多种来源构建other类，让模型见识丰富的非滴答声
        50% 概率：安全 other 文件片段（背景说话声、音乐、环境噪音等）
        30% 概率：自定义背景音（如果可用）
        20% 概率：低强度白噪声（保证最差情况也不像滴答声）
        """
        r = np.random.random()

        # 方案A：使用安全 other 文件（50%）
        if r < 0.5 and len(self.other_files_safe) > 0:
            max_attempts = 10
            for _ in range(max_attempts):
                fname = np.random.choice(self.other_files_safe)
                path = os.path.join(config.ESC_50_AUDIO_DIR, fname)
                try:
                    audio, sr = sf.read(path)
                except:
                    continue
                if sr != config.SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if len(audio) < config.SEGMENT_SIZE:
                    continue
                start = np.random.randint(0, len(audio) - config.SEGMENT_SIZE)
                seg = audio[start:start + config.SEGMENT_SIZE].astype(np.float32)
                max_val = np.max(np.abs(seg))
                if max_val > 0:
                    seg = seg / max_val
                # 加入一点背景噪声让片段更真实
                seg = seg + 0.003 * np.random.randn(len(seg)).astype(np.float32)
                return np.clip(seg, -1.0, 1.0), self.other_idx

        # 方案B：自定义背景音（30%）
        elif r < 0.8 and len(self.background_pool) > 0:
            full_audio = self.background_pool[np.random.randint(len(self.background_pool))]
            if len(full_audio) > config.SEGMENT_SIZE:
                start = np.random.randint(0, len(full_audio) - config.SEGMENT_SIZE)
                seg = full_audio[start:start + config.SEGMENT_SIZE]
            else:
                seg = full_audio.copy()
            seg = seg + 0.003 * np.random.randn(len(seg)).astype(np.float32)
            return np.clip(seg, -1.0, 1.0), self.other_idx

        # 方案C：低强度白噪声（20%，保证最差情况）
        else:
            fake = np.random.randn(config.SEGMENT_SIZE).astype(np.float32) * 0.02
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
            ds = ds.shuffle(buffer_size=min(1000, len(target_audios) + num_other))

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
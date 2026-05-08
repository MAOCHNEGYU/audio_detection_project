"""
对数梅尔谱图 (Log-Mel Spectrogram) 提取器。
端侧优化：预计算滤波器组，使用 numpy rfft，支持低频裁剪和动态范围压缩。
输入为 int16 数组，输出为 (帧数, 梅尔频带数) 的 float32 对数幅度谱。
"""

import numpy as np
import config


class MelSpectrogram:
    """
    轻量级对数梅尔谱特征提取器，专为树莓派等 ARM 设备优化。
    """
    def __init__(self,
                 sample_rate=config.SAMPLE_RATE,
                 n_fft=config.N_FFT,
                 win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH,
                 n_mels=config.N_MELS,
                 fmin=config.FMIN,
                 fmax=config.FMAX,
                 low_freq_cut=config.LOW_FREQ_CUT,
                 top_db=config.TOP_DB):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.low_freq_cut = low_freq_cut
        self.top_db = top_db

        # 预计算汉宁窗
        self.window = np.hanning(win_length).astype(np.float32)

        # 预计算梅尔滤波器组矩阵，形状 (n_mels_effective, n_fft//2+1)
        self.mel_basis = self._create_mel_filterbank()
        self.eff_n_mels = self.mel_basis.shape[0]

    def _create_mel_filterbank(self):
        """
        生成三角梅尔滤波器组。
        返回形状 (实际梅尔频带数, FFT 频点数) 的矩阵。
        """
        def mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)
        def inv_mel(m):
            return 700.0 * (10.0**(m / 2595.0) - 1.0)

        mel_low = mel(self.fmin)
        mel_high = mel(self.fmax)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        freq_points = inv_mel(mel_points)

        # 对应 FFT 频点索引
        freq_bins = np.floor((self.n_fft + 1) * freq_points / self.sr).astype(np.int32)
        fft_bins = self.n_fft // 2 + 1
        filters = np.zeros((self.n_mels, fft_bins), dtype=np.float32)

        for i in range(1, self.n_mels + 1):
            left, center, right = freq_bins[i-1], freq_bins[i], freq_bins[i+1]
            # 上升部分
            if left < center:
                slope = 1.0 / (center - left)
                filters[i-1, left:center] = slope * np.arange(center - left, dtype=np.float32)
            # 下降部分
            if center < right:
                slope = 1.0 / (right - center)
                filters[i-1, center:right] = 1.0 - slope * np.arange(right - center, dtype=np.float32)

        # 低频裁剪
        if self.low_freq_cut is not None and 0 < self.low_freq_cut < self.n_mels:
            filters = filters[self.low_freq_cut:, :]
        return filters

    def _stft(self, samples):
        """计算幅度谱，返回 (帧数, FFT频点数) 的 float32 数组。"""
        num_frames = 1 + (len(samples) - self.win_length) // self.hop_length
        if num_frames <= 0:
            return np.empty((0, self.n_fft//2+1), dtype=np.float32)

        # 手工分帧（避免引入 scipy 依赖）
        frames = np.zeros((num_frames, self.win_length), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_length
            frames[i] = samples[start:start+self.win_length]

        # 加窗
        frames *= self.window

        # 实数 FFT，自动取前 n_fft//2+1 个频点
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)
        mag = np.abs(spec)
        return mag

    def compute(self, audio):
        """
        主接口：输入 1 秒 int16 音频数组，返回对数梅尔谱。
        Args:
            audio: np.ndarray, shape (N,), dtype=int16
        Returns:
            log_mel: np.ndarray, shape (帧数, 梅尔频带数), dtype=float32
        """
        # 归一化到 [-1, 1]
        x = audio.astype(np.float32) / 32768.0

        mag_spec = self._stft(x)
        if mag_spec.shape[0] == 0:
            return np.empty((0, self.eff_n_mels), dtype=np.float32)

        # 功率谱
        power_spec = mag_spec ** 2
        # 梅尔滤波
        mel_energy = np.dot(power_spec, self.mel_basis.T)
        # 对数域 + 防零
        log_mel = np.log(mel_energy + 1e-6)

        # 动态范围裁剪
        if self.top_db is not None:
            max_val = np.max(log_mel)
            if max_val != 0:
                log_mel = np.maximum(log_mel, max_val - self.top_db)

        return log_mel.astype(np.float32)
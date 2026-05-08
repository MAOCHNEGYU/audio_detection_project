"""
MFCC 提取器 (Mel-Frequency Cepstral Coefficients)。
在 Log-Mel 基础上通过 DCT 去相关、压缩维度，常用于语音识别和轻量分类器。
"""

import numpy as np
from features.mel_spectrogram import MelSpectrogram
import config


class MFCC:
    """
    MFCC 特征提取器，复用 MelSpectrogram 中的 STFT 和梅尔滤波逻辑，
    额外添加 DCT 步骤。
    """
    def __init__(self,
                 sample_rate=config.SAMPLE_RATE,
                 n_fft=config.N_FFT,
                 win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH,
                 n_mels=config.N_MELS,
                 n_mfcc=config.N_MFCC,
                 fmin=config.FMIN,
                 fmax=config.FMAX,
                 low_freq_cut=config.LOW_FREQ_CUT):
        self.n_mfcc = n_mfcc
        # 内部使用 MelSpectrogram 完成前面步骤
        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            low_freq_cut=low_freq_cut,
            top_db=None  # MFCC 一般在 DCT 前不做 top_db
        )
        self.eff_n_mels = self.mel_extractor.eff_n_mels
        # 预计算 DCT 矩阵 (n_mfcc, effective_mels)
        self.dct_matrix = self._create_dct_matrix()

    def _create_dct_matrix(self):
        """生成 DCT-II 型变换矩阵。"""
        n = np.arange(self.eff_n_mels)
        k = np.arange(self.n_mfcc)[:, None]
        mat = np.cos(np.pi * (n + 0.5) * k / self.eff_n_mels)
        return mat.astype(np.float32)

    def compute(self, audio):
        """
        输入 int16 音频，输出 (帧数, n_mfcc) 的 MFCC 矩阵。
        """
        # 先得到 log-mel (不裁剪 top_db)
        log_mel = self.mel_extractor.compute(audio)
        if log_mel.shape[0] == 0:
            return np.empty((0, self.n_mfcc), dtype=np.float32)
        # DCT 变换
        mfcc = np.dot(log_mel, self.dct_matrix.T)
        return mfcc.astype(np.float32)
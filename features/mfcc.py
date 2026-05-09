"""MFCC提取器"""
import numpy as np
from features.mel_spectrogram import MelSpectrogram
import config

class MFCC:
    def __init__(self, sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT, win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, n_mfcc=config.N_MFCC,
                 fmin=config.FMIN, fmax=config.FMAX, low_freq_cut=config.LOW_FREQ_CUT):
        self.n_mfcc = n_mfcc
        self.mel_extractor = MelSpectrogram(sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax, low_freq_cut, top_db=None)
        self.eff_n_mels = self.mel_extractor.eff_n_mels
        self.dct_matrix = self._create_dct_matrix()

    def _create_dct_matrix(self):
        n = np.arange(self.eff_n_mels)
        k = np.arange(self.n_mfcc)[:, None]
        mat = np.cos(np.pi * (n + 0.5) * k / self.eff_n_mels)
        return mat.astype(np.float32)

    def compute(self, audio):
        log_mel = self.mel_extractor.compute(audio)
        if log_mel.shape[0] == 0:
            return np.empty((0, self.n_mfcc), dtype=np.float32)
        mfcc = np.dot(log_mel, self.dct_matrix.T)
        return mfcc.astype(np.float32)
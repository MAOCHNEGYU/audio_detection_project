"""对数梅尔谱图提取器"""
import numpy as np
import config

class MelSpectrogram:
    def __init__(self, sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT, win_length=config.WIN_LENGTH,
                 hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX,
                 low_freq_cut=config.LOW_FREQ_CUT, top_db=config.TOP_DB):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.low_freq_cut = low_freq_cut
        self.top_db = top_db
        self.window = np.hanning(win_length).astype(np.float32)
        self.mel_basis = self._create_mel_filterbank()
        self.eff_n_mels = self.mel_basis.shape[0]

    def _create_mel_filterbank(self):
        def mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def inv_mel(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)
        mel_low = mel(self.fmin)
        mel_high = mel(self.fmax)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        freq_points = inv_mel(mel_points)
        freq_bins = np.floor((self.n_fft + 1) * freq_points / self.sr).astype(np.int32)
        fft_bins = self.n_fft // 2 + 1
        filters = np.zeros((self.n_mels, fft_bins), dtype=np.float32)
        for i in range(1, self.n_mels + 1):
            left, center, right = freq_bins[i-1], freq_bins[i], freq_bins[i+1]
            if left < center:
                filters[i-1, left:center] = np.linspace(0, 1, center - left, dtype=np.float32)
            if center < right:
                filters[i-1, center:right] = np.linspace(1, 0, right - center, dtype=np.float32)
        if self.low_freq_cut is not None and 0 < self.low_freq_cut < self.n_mels:
            filters = filters[self.low_freq_cut:, :]
        return filters

    def _stft(self, samples):
        num_frames = 1 + (len(samples) - self.win_length) // self.hop_length
        if num_frames <= 0:
            return np.empty((0, self.n_fft//2+1), dtype=np.float32)
        frames = np.zeros((num_frames, self.win_length), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_length
            frames[i] = samples[start:start+self.win_length]
        frames *= self.window
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)
        return np.abs(spec)

    def compute(self, audio):
        x = audio.astype(np.float32) / 32768.0
        mag_spec = self._stft(x)
        if mag_spec.shape[0] == 0:
            return np.empty((0, self.eff_n_mels), dtype=np.float32)
        power_spec = mag_spec ** 2
        mel_energy = np.dot(power_spec, self.mel_basis.T)
        log_mel = np.log(mel_energy + 1e-6)
        if self.top_db is not None:
            max_val = np.max(log_mel)
            if max_val != 0:
                log_mel = np.maximum(log_mel, max_val - self.top_db)
        return log_mel.astype(np.float32)
"""原始波形特征"""
import numpy as np
import config

class RawWaveform:
    def __init__(self, target_length=config.RAW_TARGET_LENGTH, normalize=True):
        self.target_length = target_length
        self.normalize = normalize

    def compute(self, audio):
        sig = audio.astype(np.float32)
        if self.normalize:
            sig /= 32768.0
        if len(sig) >= self.target_length:
            return sig[:self.target_length].astype(np.float32)
        else:
            return np.pad(sig, (0, self.target_length - len(sig)), mode='constant').astype(np.float32)

class RawWaveformFrames:
    def __init__(self, frame_len=config.RAW_FRAME_LEN, hop_len=config.RAW_HOP_LEN, normalize=True):
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.normalize = normalize

    def compute(self, audio):
        sig = audio.astype(np.float32)
        if self.normalize:
            sig /= 32768.0
        num_frames = 1 + (len(sig) - self.frame_len) // self.hop_len
        if num_frames <= 0:
            return np.empty((0, self.frame_len), dtype=np.float32)
        frames = np.zeros((num_frames, self.frame_len), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_len
            frames[i] = sig[start:start+self.frame_len]
        return frames
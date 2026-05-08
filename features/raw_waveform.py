"""
原始波形特征提取器。
提供两种模式：
1. RawWaveform: 将音频重采样/截取到固定长度，直接作为 1D 卷积输入。
2. RawWaveformFrames: 将音频切分成重叠的时域帧，适合浅层 1D CNN 学习局部模式。
"""

import numpy as np
import config


class RawWaveform:
    """
    将音频规整到固定长度，数值归一化到 [-1, 1]。
    """
    def __init__(self, target_length=config.RAW_TARGET_LENGTH, normalize=True):
        self.target_length = target_length
        self.normalize = normalize

    def compute(self, audio):
        """
        Args:
            audio: np.ndarray, shape (N,), dtype=int16
        Returns:
            np.ndarray, shape (target_length,), dtype=float32
        """
        sig = audio.astype(np.float32)
        if self.normalize:
            sig /= 32768.0

        # 长度调整：大于目标则截断，小于则补零
        if len(sig) >= self.target_length:
            result = sig[:self.target_length]
        else:
            pad = self.target_length - len(sig)
            result = np.pad(sig, (0, pad), mode='constant')
        return result.astype(np.float32)


class RawWaveformFrames:
    """
    将音频切分成重叠的短帧，每帧作为 1D 卷积的一个时间片。
    输出形状 (帧数, 帧长)。
    """
    def __init__(self, frame_len=config.RAW_FRAME_LEN,
                 hop_len=config.RAW_HOP_LEN, normalize=True):
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.normalize = normalize

    def compute(self, audio):
        """
        Args:
            audio: np.ndarray, shape (N,), dtype=int16
        Returns:
            np.ndarray, shape (num_frames, frame_len), dtype=float32
        """
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
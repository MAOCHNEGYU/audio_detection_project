"""
真实滴答声数据加载器
v1.3: 从录制文件中切出目标片段，用于微调
"""
import numpy as np
import soundfile as sf
import config

class RealClockDataset:
    def __init__(self, wav_path=config.REAL_CLOCK_PATH):
        self.audio, self.sr = sf.read(wav_path)
        if self.sr != config.SAMPLE_RATE:
            raise ValueError(f"真实音频采样率必须为 {config.SAMPLE_RATE}Hz，当前为 {self.sr}Hz")
        if self.audio.ndim > 1:
            self.audio = self.audio.mean(axis=1)
        self.segments = self._split()

    def _split(self):
        segs = []
        seg_len = config.SEGMENT_SIZE
        step = seg_len // 2  # 0.5 秒重叠
        for start in range(0, len(self.audio) - seg_len + 1, step):
            seg = self.audio[start:start+seg_len].astype(np.float32)
            max_val = np.max(np.abs(seg))
            if max_val > 0:
                seg = seg / max_val
            segs.append(seg)
        return segs

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], 0  # 标签 0 = clock_tick
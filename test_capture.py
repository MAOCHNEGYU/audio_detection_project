"""
测试当前项目的音频采集模块是否正常工作。
"""
import time
import numpy as np
from audio.capture import AudioCapture
from utils.logger import log_info

def on_segment(segment, timestamp):
    rms = np.sqrt(np.mean(segment.astype(np.float32) ** 2))
    log_info(f"收到音频片段 | 形状: {segment.shape} | RMS: {rms:.1f}")

log_info("开始测试音频采集...")
cap = AudioCapture()
cap.start(on_segment)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    log_info("测试结束")
finally:
    cap.stop()
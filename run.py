"""
主程序入口：演示音频采集 + 特征提取 + 模型推理 + 告警的端到端流水线。
在树莓派上运行后，每 1 秒输出预测的事件类别和概率。
可按 Ctrl+C 停止。
"""

import time
import numpy as np
import os
from audio.capture import AudioCapture
from features.mel_spectrogram import MelSpectrogram
from models.inference import TFLiteInference
from utils.logger import log_info, log_warn, log_error
import config

_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, _mel.eff_n_mels, 1)

# ---------- 初始化各模块 ----------
mel_extractor = MelSpectrogram()
# 如果 TFLite 模型已存在，则加载推理器
inference_engine = None
if os.path.exists(config.TFLITE_MODEL_PATH):
    inference_engine = TFLiteInference(config.TFLITE_MODEL_PATH)
    log_info("TFLite 模型加载成功")
else:
    log_warn(f"未找到 TFLite 模型 {config.TFLITE_MODEL_PATH}，仅演示特征提取")


def on_segment_ready(audio_segment, timestamp):
    """
    每获取到一个 1 秒音频段时的回调函数。
    完成特征提取、模型推理（如可用），并模拟告警。
    """
    # 1. 提取 Log-Mel 特征
    log_mel = mel_extractor.compute(audio_segment)   # (时间帧, 32)
    # 确保形状符合模型输入：需要填充/截断到固定时间帧 (49 帧)
    target_frames = config.MODEL_INPUT_SHAPE[0]  # 49
    if log_mel.shape[0] < target_frames:
        # 零填充
        pad_frames = target_frames - log_mel.shape[0]
        log_mel = np.vstack([log_mel, np.zeros((pad_frames, log_mel.shape[1]), dtype=np.float32)])
    elif log_mel.shape[0] > target_frames:
        log_mel = log_mel[:target_frames, :]

    # 添加通道维度: (49, 32) -> (49, 32, 1) -> (1, 49, 32, 1)
    model_input = np.expand_dims(log_mel, axis=-1)

    # 2. 推理
    if inference_engine:
        try:
            probs = inference_engine.predict(model_input)
            # 获取预测类别（阈值 0.5）
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            event_name = config.CLASS_NAMES[predicted_class]
            # 阈值
            if confidence >= 0.5:
                log_info(f"检测到事件: {event_name}, 置信度: {probs.round(3)}")
                # TODO: 触发 GPIO 告警（LED/蜂鸣器）
            else:
                log_info("无事件触发")
        except Exception as e:
            log_error(f"推理失败: {e}")
    else:
        # 仅演示特征形状
        log_info(f"特征形状: {log_mel.shape}")


if __name__ == "__main__":
    log_info("启动端侧音频检测系统原型...")
    cap = AudioCapture()
    cap.start(on_segment_ready)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log_info("用户中断")
    finally:
        cap.stop()
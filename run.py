"""
实时音频检测（多分类，含 other）。
带自动增益控制与能量门控，防止安静环境误判。
每次推理的音频自动保存到 debug_audio.wav，方便离线对比。
"""
import time
import numpy as np
import os
import sys
import wave
import json
from audio.capture import AudioCapture
from features.mel_spectrogram import MelSpectrogram
from models.inference import TFLiteInference
from utils.logger import log_info, log_warn, log_error
from utils.calibration import calibrate_threshold, calibrate_energy_threshold
import config

# ---------- 初始化 ----------
_mel = MelSpectrogram()
config.MODEL_INPUT_SHAPE = (49, _mel.eff_n_mels, 1)

mel_extractor = MelSpectrogram()
inference_engine = None
if os.path.exists(config.TFLITE_MODEL_PATH):
    inference_engine = TFLiteInference(config.TFLITE_MODEL_PATH)
    log_info("TFLite 模型加载成功")
else:
    log_warn("TFLite 模型未找到，仅运行特征提取")

# ---------- 动态阈值加载/校准 ----------
dynamic_msp_threshold = None
dynamic_energy_threshold = None
if config.USE_DYNAMIC_THRESHOLD and inference_engine:
    if config.OOD_METHOD == "msp":
        msp_valid = False
        if os.path.exists(config.THRESHOLD_SAVE_PATH):
            try:
                with open(config.THRESHOLD_SAVE_PATH) as f:
                    data = json.load(f)
                    dynamic_msp_threshold = data['threshold']
                    log_info(f"已加载 MSP 动态阈值: {dynamic_msp_threshold:.4f}")
                    msp_valid = True
            except (json.JSONDecodeError, KeyError) as e:
                log_warn(f"阈值文件损坏 ({e})，将重新校准")
                os.remove(config.THRESHOLD_SAVE_PATH)
        if not msp_valid:
            log_info("开始 MSP 环境校准...")
            log_info("请保持环境安静，避免出现目标事件...")
            dynamic_msp_threshold = calibrate_threshold(
                inference_engine, mel_extractor,
                duration_sec=config.ENV_CALIBRATION_SECONDS
            )
            os.makedirs(os.path.dirname(config.THRESHOLD_SAVE_PATH), exist_ok=True)
            with open(config.THRESHOLD_SAVE_PATH, 'w') as f:
                json.dump({'threshold': float(dynamic_msp_threshold)}, f)
            log_info(f"MSP 动态阈值已保存: {dynamic_msp_threshold:.4f}")

    elif config.OOD_METHOD == "energy":
        energy_valid = False
        if os.path.exists(config.ENERGY_CALIB_PATH):
            try:
                with open(config.ENERGY_CALIB_PATH) as f:
                    data = json.load(f)
                    dynamic_energy_threshold = data['threshold']
                    log_info(f"已加载能量动态阈值: {dynamic_energy_threshold:.4f}")
                    energy_valid = True
            except (json.JSONDecodeError, KeyError):
                log_warn("能量阈值文件损坏，将重新校准")
                os.remove(config.ENERGY_CALIB_PATH)
        if not energy_valid:
            log_info("开始能量分数环境校准...")
            dynamic_energy_threshold = calibrate_energy_threshold(
                inference_engine, mel_extractor,
                duration_sec=config.ENV_CALIBRATION_SECONDS
            )
            with open(config.ENERGY_CALIB_PATH, 'w') as f:
                json.dump({'threshold': float(dynamic_energy_threshold)}, f)
            log_info(f"能量动态阈值已保存: {dynamic_energy_threshold:.4f}")

# ---------- 核心回调 ----------
def on_segment_ready(audio_segment, timestamp):
    rms = np.sqrt(np.mean(audio_segment.astype(np.float32)**2))
    if rms < 1200:
        log_info(f"低能量跳过 (RMS={rms:.1f})")
        return

    # 自动增益控制
    target_rms = 3000.0
    if rms > 1e-6:
        gain = target_rms / rms
        audio_segment = (audio_segment.astype(np.float32) * gain).astype(np.int16)

    # ----- 每次都保存，覆盖旧文件，方便随时用 test_debug.py 测试 -----
    wav_path = "debug_audio.wav"
    try:
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(audio_segment.tobytes())
    except Exception as e:
        log_error(f"保存调试音频失败: {e}")

    # 特征提取
    log_mel = mel_extractor.compute(audio_segment)
    target_frames = config.MODEL_INPUT_SHAPE[0]
    if log_mel.shape[0] < target_frames:
        pad = np.zeros((target_frames - log_mel.shape[0], log_mel.shape[1]), dtype=np.float32)
        log_mel = np.vstack([log_mel, pad])
    elif log_mel.shape[0] > target_frames:
        log_mel = log_mel[:target_frames, :]
    model_input = np.expand_dims(log_mel, axis=-1)

    if inference_engine is None:
        log_info(f"特征形状: {log_mel.shape}")
        return

    probs = inference_engine.predict(model_input)

    # 根据配置方法判断
    if config.OOD_METHOD == "msp":
        threshold = dynamic_msp_threshold if (config.USE_DYNAMIC_THRESHOLD and dynamic_msp_threshold) else config.FIXED_CONFIDENCE_THRESHOLD
        conf = np.max(probs)
        if conf >= threshold:
            pred = np.argmax(probs)
            event = config.CLASS_NAMES[pred]
        else:
            event = "other"
    elif config.OOD_METHOD == "energy":
        energy = -np.log(np.max(probs) + 1e-10)
        threshold = dynamic_energy_threshold if (config.USE_DYNAMIC_THRESHOLD and dynamic_energy_threshold) else (config.ENERGY_THRESHOLD or 2.0)
        if energy <= threshold:
            pred = np.argmax(probs)
            event = config.CLASS_NAMES[pred]
        else:
            event = "other"
    else:
        pred = np.argmax(probs)
        event = config.CLASS_NAMES[pred]

    prob_str = " ".join([f"{config.CLASS_NAMES[i]}:{probs[i]:.3f}" for i in range(len(probs))])
    log_info(f"预测: {event}  |  [{prob_str}]")

if __name__ == "__main__":
    log_info("启动实时检测（带能量门控与 AGC）... 音频会自动保存为 debug_audio.wav")
    cap = AudioCapture()
    cap.start(on_segment_ready)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log_info("用户中断")
    finally:
        cap.stop()
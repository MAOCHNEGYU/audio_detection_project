"""
环境校准：采集背景音计算自适应阈值（MSP / 能量）。
"""
import numpy as np
import sounddevice as sd
import config

def calibrate_threshold(inference_engine, mel_extractor, duration_sec=15):
    """
    录制背景音，计算 Softmax 最大概率的均值 + k*std 作为 MSP 阈值。
    """
    sr = config.SAMPLE_RATE
    segment_size = config.SEGMENT_SIZE
    print(f"正在录制 {duration_sec} 秒环境音频用于 MSP 校准...")
    recording = sd.rec(
        int(duration_sec * sr), samplerate=sr, channels=1, dtype='int16'
    )
    sd.wait()
    step = segment_size // 2
    max_probs = []
    for start in range(0, len(recording) - segment_size + 1, step):
        seg = recording[start:start+segment_size].flatten().astype(np.int16)
        log_mel = mel_extractor.compute(seg)
        target_frames = config.MODEL_INPUT_SHAPE[0]
        if log_mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - log_mel.shape[0], log_mel.shape[1]), dtype=np.float32)
            log_mel = np.vstack([log_mel, pad])
        else:
            log_mel = log_mel[:target_frames]
        model_input = np.expand_dims(log_mel, axis=-1)
        probs = inference_engine.predict(model_input)
        max_probs.append(np.max(probs))
    if len(max_probs) == 0:
        print("警告：未收集到有效片段，使用默认阈值 0.7")
        return 0.7
    mu = np.mean(max_probs)
    sigma = np.std(max_probs)
    threshold = mu + config.THRESHOLD_STD_MULTIPLIER * sigma
    print(f"MSP 校准完成：均值={mu:.4f}, 标准差={sigma:.4f}, 阈值={threshold:.4f}")
    return float(threshold)

def calibrate_energy_threshold(inference_engine, mel_extractor, duration_sec=15):
    """
    录制背景音，计算能量分数 (-log max prob) 的均值 + k*std 作为能量阈值。
    """
    sr = config.SAMPLE_RATE
    segment_size = config.SEGMENT_SIZE
    print(f"正在录制 {duration_sec} 秒环境音频用于能量校准...")
    recording = sd.rec(
        int(duration_sec * sr), samplerate=sr, channels=1, dtype='int16'
    )
    sd.wait()
    step = segment_size // 2
    energies = []
    for start in range(0, len(recording) - segment_size + 1, step):
        seg = recording[start:start+segment_size].flatten().astype(np.int16)
        log_mel = mel_extractor.compute(seg)
        target_frames = config.MODEL_INPUT_SHAPE[0]
        if log_mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - log_mel.shape[0], log_mel.shape[1]), dtype=np.float32)
            log_mel = np.vstack([log_mel, pad])
        else:
            log_mel = log_mel[:target_frames]
        model_input = np.expand_dims(log_mel, axis=-1)
        probs = inference_engine.predict(model_input)
        energy = -np.log(np.max(probs) + 1e-10)
        energies.append(energy)
    if len(energies) == 0:
        print("警告：未收集到有效片段，使用默认能量阈值 2.0")
        return 2.0
    mu = np.mean(energies)
    sigma = np.std(energies)
    threshold = mu + config.THRESHOLD_STD_MULTIPLIER * sigma
    print(f"能量校准完成：均值={mu:.4f}, 标准差={sigma:.4f}, 阈值={threshold:.4f}")
    return float(threshold)
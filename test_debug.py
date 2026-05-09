"""
离线测试 debug_audio.wav，模拟实时处理流程，检查模型对实际播放音频的识别能力。
"""
import numpy as np
import soundfile as sf
import config
from features.mel_spectrogram import MelSpectrogram
from models.inference import TFLiteInference

def test_wav(wav_path="debug_audio.wav"):
    # 读取音频
    audio, sr = sf.read(wav_path)
    if sr != config.SAMPLE_RATE:
        raise ValueError(f"采样率应为 {config.SAMPLE_RATE}Hz，实际 {sr}Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 长度对齐
    if len(audio) < config.SEGMENT_SIZE:
        pad_len = config.SEGMENT_SIZE - len(audio)
        audio = np.pad(audio, (0, pad_len), mode='constant')
    else:
        audio = audio[:config.SEGMENT_SIZE]

    # 转为 int16（模拟实时采集格式）
    audio_int16 = (audio * 32767).astype(np.int16)

    # 提取特征
    mel_extractor = MelSpectrogram()
    log_mel = mel_extractor.compute(audio_int16)
    target_frames = config.MODEL_INPUT_SHAPE[0]
    if log_mel.shape[0] < target_frames:
        pad = np.zeros((target_frames - log_mel.shape[0], log_mel.shape[1]), dtype=np.float32)
        log_mel = np.vstack([log_mel, pad])
    else:
        log_mel = log_mel[:target_frames]
    model_input = np.expand_dims(log_mel, axis=-1)

    # 加载 TFLite 模型并推理
    infer = TFLiteInference(config.TFLITE_MODEL_PATH)
    probs = infer.predict(model_input)
    pred_class = np.argmax(probs)
    prob_str = " ".join([f"{config.CLASS_NAMES[i]}:{probs[i]:.3f}" for i in range(len(probs))])
    print(f"离线测试结果: 预测类别 = {config.CLASS_NAMES[pred_class]}")
    print(f"概率分布: {prob_str}")

if __name__ == "__main__":
    test_wav()
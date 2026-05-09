"""
量化对比实验：Keras FP32 vs. TFLite FP32 vs. TFLite INT8。
"""

import os, time
import numpy as np
import tensorflow as tf
from utils.project_init import set_project_root
set_project_root()

import config
from data.dataset import ESC50Dataset
from models.inference import TFLiteInference

# 加载验证数据（少量用于测试）
val_ds = ESC50Dataset(fold=0, mode='val', augment=False).dataset
X_val, y_val = [], []
for x, y in val_ds.take(10):   # 少量样本
    X_val.append(x.numpy())
    y_val.append(y.numpy())
X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)

# 原始 Keras 模型
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
start = time.time()
y_pred_keras = model.predict(X_val, verbose=0)
keras_time = time.time() - start
keras_acc = np.mean(np.argmax(y_pred_keras, axis=1) == np.argmax(y_val, axis=1))

# TFLite FP32
infer_tflite = TFLiteInference(config.TFLITE_MODEL_PATH)
start = time.time()
y_pred_tflite = [infer_tflite.predict(X_val[i]) for i in range(len(X_val))]
tflite_time = time.time() - start
tflite_acc = np.mean(np.argmax(y_pred_tflite, axis=1) == np.argmax(y_val, axis=1))

print(f"Keras  acc={keras_acc:.4f}, time={keras_time:.4f}s")
print(f"TFLite acc={tflite_acc:.4f}, time={tflite_time:.4f}s")

# INT8 量化（如果启用）
if config.QUANTIZATION_ENABLED:
    # 构建代表数据集并转换
    def representative_dataset():
        for i in range(10):
            yield [X_val[i:i+1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8_model = converter.convert()
    with open("models/saved/audio_detector_int8.tflite", 'wb') as f:
        f.write(tflite_int8_model)
    print(f"INT8 模型大小: {len(tflite_int8_model)/1024:.1f} KB")
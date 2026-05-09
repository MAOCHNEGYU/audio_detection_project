"""
TFLite 模型推理器封装。
加载 .tflite 模型，提供 predict() 方法，输入 log-mel 特征张量，输出概率数组。
"""
import numpy as np
import tensorflow as tf
import config


class TFLiteInference:
    def __init__(self, model_path=config.TFLITE_MODEL_PATH):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']

    def predict(self, features):
        # 自动扩展 batch 维度
        if features.ndim == 3:
            features = np.expand_dims(features, axis=0)
        elif features.ndim == 2:
            features = features[np.newaxis, :, :, np.newaxis]
        elif features.ndim == 4 and features.shape[0] != 1:
            raise ValueError("仅支持单样本推理")
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]
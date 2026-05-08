"""
TFLite 模型推理器封装。
加载 .tflite 模型，提供 predict() 方法，输入 log-mel 特征张量，输出概率数组。
设计为在树莓派上使用 tflite-runtime 高效运行。
"""

import numpy as np
import tensorflow as tf   # 或使用 tflite_runtime.interpreter
import config


class TFLiteInference:
    """
    TensorFlow Lite 推理器。
    初始化时加载模型，分配张量。
    """
    def __init__(self, model_path=config.TFLITE_MODEL_PATH):
        self.model_path = model_path
        # 加载模型
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # 获取输入/输出细节
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 记录输入形状（用于检查特征是否匹配）
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']

    def predict(self, features):
        """
        执行单次推理。
        Args:
            features: np.ndarray, 形状 (1, 时间帧, 梅尔频带, 1) 或 (时间帧, 梅尔频带, 1) 或 (时间帧, 梅尔频带)
        Returns:
            probs: np.ndarray, 形状 (num_classes,)，各类别概率 [0,1]
        """
        # 自动扩展 batch 维度
        if features.ndim == 3:
            features = np.expand_dims(features, axis=0)
        elif features.ndim == 2:
            features = features[np.newaxis, :, :, np.newaxis]
        elif features.ndim == 4 and features.shape[0] != 1:
            raise ValueError("仅支持单样本推理")

        # 确保数据类型为 float32
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # 设置输入张量
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        # 运行推理
        self.interpreter.invoke()
        # 获取输出
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]  # 去除 batch 维度


if __name__ == "__main__":
    # 测试加载模型（需要在训练后才存在）
    import os
    if os.path.exists(config.TFLITE_MODEL_PATH):
        infer = TFLiteInference()
        # 构造随机输入进行测试
        dummy = np.random.randn(1, *config.MODEL_INPUT_SHAPE).astype(np.float32)
        probs = infer.predict(dummy)
        print(f"预测概率: {probs}")
    else:
        print("模型文件不存在，请先运行 train.py 生成模型。")
"""
轻量化音频事件检测模型定义。
基于深度可分离卷积（MobileNetV3 风格瓶颈）+ 时序 GRU，输出多标签概率。
参考结构：2D CNN 提取频域特征 → GRU 建模时序 → Sigmoid 多标签分类。
参数量可通过宽度系数 alpha 控制，默认总参数量 < 1M。
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import config


def _bottleneck_block(inputs, in_channels, out_channels, expansion_factor, stride, alpha):
    """
    MobileNetV3 轻量瓶颈块（使用深度可分离卷积 + 线性瓶颈 + SE 可选）。
    为简单起见，此处省略 SE 模块（可进一步压缩参数），保留核心的高效结构。
    """
    exp_channels = int(in_channels * expansion_factor)
    # 1x1 扩张
    x = layers.Conv2D(exp_channels, 1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3x3 深度可分离卷积
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1x1 投影，线性激活（无ReLU）
    x = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 残差连接（当输入/输出形状一致且步长为1时）
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([x, inputs])
    return x


def create_model(input_shape=config.MODEL_INPUT_SHAPE,
                 num_classes=config.NUM_CLASSES,
                 alpha=config.MODEL_WIDTH_MULTIPLIER,
                 gru_units=config.GRU_UNITS,
                 dropout=config.MODEL_DROPOUT):
    """
    构建轻量 CRNN 模型。
    Args:
        input_shape: 输入特征图形状 (时间帧, 梅尔频带, 1)
        num_classes: 输出事件类别数
        alpha: 宽度乘数（<1 削减通道，降低参数量）
        gru_units: GRU 隐藏单元数
        dropout: Dropout 概率
    Returns:
        tf.keras.Model 实例
    """
    inputs = layers.Input(shape=input_shape, name='log_mel')

    # ---- 卷积前端，逐步下采样频率维度 ----
    # Block 1: 初始卷积，下采样时间维度
    x = layers.Conv2D(int(16 * alpha), 3, strides=(2, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2: 瓶颈块，保持尺寸
    x = _bottleneck_block(x, int(16 * alpha), int(16 * alpha), expansion_factor=3, stride=1, alpha=alpha)

    # Block 3: 下采样频率和时间
    x = _bottleneck_block(x, int(16 * alpha), int(24 * alpha), expansion_factor=3, stride=(2, 2), alpha=alpha)

    # Block 4: 保持尺寸
    x = _bottleneck_block(x, int(24 * alpha), int(24 * alpha), expansion_factor=3, stride=1, alpha=alpha)

    # Block 5: 再次下采样频率，保留时间
    x = _bottleneck_block(x, int(24 * alpha), int(32 * alpha), expansion_factor=3, stride=(2, 1), alpha=alpha)

    # Block 6: 最终下采样频率（输出时间轴长度 = 原始帧数 / (2*2) ≈ 12）
    x = _bottleneck_block(x, int(32 * alpha), int(48 * alpha), expansion_factor=3, stride=(2, 1), alpha=alpha)

    # ---- 准备输入 GRU：将频率和通道合并为特征向量 ----
    # x 形状: (batch, time, freq, channels)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)   # (batch, time, features)

    # ---- 时序建模 ----
    x = layers.GRU(gru_units, return_sequences=False, dropout=dropout)(x)

    # ---- 分类头 ----
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LightAudioCRNN')
    return model

# ---------- 在文件末尾追加以下内容 ----------

def create_tiny_model(input_shape=config.MODEL_INPUT_SHAPE,
                      num_classes=config.NUM_CLASSES,
                      alpha=0.25,
                      dropout=0.5,
                      l2_reg=1e-4):
    """
    极轻量模型：三层深度可分离卷积 + 全局平均池化 + 全连接。
    无 GRU，避免时序过拟合；使用 L2 正则化和较大 Dropout。
    默认总参数量 < 50K。
    """
    reg = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name='log_mel')

    # Block 1
    x = layers.SeparableConv2D(int(16 * alpha), 3, strides=2, padding='same',
                               depthwise_regularizer=reg, pointwise_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2
    x = layers.SeparableConv2D(int(32 * alpha), 3, strides=2, padding='same',
                               depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 3
    x = layers.SeparableConv2D(int(64 * alpha), 3, strides=2, padding='same',
                               depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 全局池化
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg,
                           name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TinyAudioCNN')
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
    print(f"总参数量: {model.count_params():,}")
    # 预期 < 1,000,000
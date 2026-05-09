"""
轻量化模型定义（纯 CNN，无 GRU）。
v1.1: alpha 临时调整为 0.8，增大容量
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import config


def create_tiny_model(input_shape=None,
                      num_classes=config.NUM_CLASSES,
                      alpha=0.8,           # 原 0.5 -> 0.8
                      dropout=0.3,
                      l2_reg=1e-4):
    if input_shape is None:
        input_shape = config.MODEL_INPUT_SHAPE
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
    x = layers.SeparableConv2D(int(64 * alpha), 3, strides=1, padding='same',
                               depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 4
    x = layers.SeparableConv2D(int(64 * alpha), 3, strides=2, padding='same',
                               depthwise_regularizer=reg, pointwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg,
                           dtype='float32', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TinyAudioCNN_v3')
    return model
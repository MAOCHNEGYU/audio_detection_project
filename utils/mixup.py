"""
频谱 Mixup 实现
"""

import numpy as np
import tensorflow as tf

def mixup_batch(X, y, alpha=0.4):
    """对一批数据应用 Mixup。
    X: (batch, time, freq, 1)
    y: (batch, num_classes) one-hot 标签
    返回混合后的 X 和 y
    """
    batch_size = tf.shape(X)[0]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    # 随机打乱同批样本索引
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_X = lam * X + (1 - lam) * tf.gather(X, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    return mixed_X, mixed_y
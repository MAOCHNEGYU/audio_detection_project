"""
GPU训练工具模块。
提供GPU检测、内存配置、混合精度策略设置等功能。
"""

import os
import tensorflow as tf
import config


def setup_gpu_environment():
    """
    配置GPU运行环境：动态内存增长、可见设备设置、混合精度。

    Returns:
        tf.distribute.Strategy: 分布式策略（单GPU为默认策略）
    """
    # 禁用TF的冗余日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 获取物理GPU列表
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("警告：未检测到GPU，将使用CPU训练。训练速度可能较慢。")
        return tf.distribute.get_strategy()

    print(f"检测到 {len(gpus)} 块GPU:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")

    # 启用动态内存增长（避免一次性占满显存，允许多实验共存）
    if config.GPU_MEMORY_GROWTH:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  已启用GPU动态内存增长")
            except RuntimeError as e:
                print(f"  GPU内存增长设置失败（可能已初始化）: {e}")

    # 设置混合精度策略（FP16计算+FP32累加，V100/RTX系列可提速2-3倍）
    if config.MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"  已启用混合精度策略: {policy.name}")

    # 启用XLA JIT编译（图优化，减少Kernel Launch开销）
    if config.XLA_JIT:
        tf.config.optimizer.set_jit(True)
        print("  已启用XLA JIT编译优化")

    # 返回默认策略（单GPU）或MirroredStrategy（多GPU）
    strategy = tf.distribute.get_strategy()
    print(f"  使用策略: {strategy.__class__.__name__}")

    return strategy


def print_gpu_memory_usage():
    """
    打印当前GPU显存使用情况（调试用）。
    需要nvidia-ml-py3包支持。
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_mb = info.used / 1024 ** 2
        total_mb = info.total / 1024 ** 2
        print(f"GPU显存使用: {used_mb:.1f}MB / {total_mb:.1f}MB ({used_mb / total_mb * 100:.1f}%)")
    except ImportError:
        print("未安装pynvml，无法监控显存。pip install nvidia-ml-py3")
    except Exception as e:
        print(f"显存读取失败: {e}")
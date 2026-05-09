"""GPU环境配置"""
import os
import tensorflow as tf
import config

def setup_gpu_environment():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("警告：未检测到GPU，使用CPU训练。")
        return tf.distribute.get_strategy()
    print(f"检测到 {len(gpus)} 块GPU:")
    for i, gpu in enumerate(gpus):
        print(f"  [{i}] {gpu.name}")
    if config.GPU_MEMORY_GROWTH:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("  已启用GPU动态内存增长")
            except RuntimeError as e:
                print(f"  内存增长设置失败: {e}")
    if config.MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"  已启用混合精度策略: {policy.name}")
    if config.XLA_JIT:
        tf.config.optimizer.set_jit(True)
        print("  已启用XLA JIT编译优化")
    strategy = tf.distribute.get_strategy()
    print(f"  使用策略: {strategy.__class__.__name__}")
    return strategy

def print_gpu_memory_usage():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_mb = info.used / 1024**2
        total_mb = info.total / 1024**2
        print(f"GPU显存使用: {used_mb:.1f}MB / {total_mb:.1f}MB ({used_mb/total_mb*100:.1f}%)")
    except ImportError:
        print("未安装pynvml，无法监控显存。pip install nvidia-ml-py3")
    except Exception as e:
        print(f"显存读取失败: {e}")
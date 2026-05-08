"""
全局配置参数
所有模块共用的采样率、帧长、特征参数等均在此定义，方便统一修改。
"""

# ========== 音频采集 ==========
SAMPLE_RATE = 16000          # 采样率 (Hz)
CHANNELS = 1                 # 声道数 (单声道)
SAMPLE_WIDTH = 2             # 位宽 (字节)，16bit = 2
FRAMES_PER_BUFFER = 512     # PortAudio 每次回调的帧数 (约32ms 音频)
SEGMENT_SECONDS = 1.0        # 每次处理的音频时长 (秒)
SEGMENT_SIZE = int(SAMPLE_RATE * SEGMENT_SECONDS)  # 16000 帧
QUEUE_MAX_SIZE = 100         # 音频队列最大缓存段数

# ========== 特征提取通用 ==========
N_FFT = 512                  # FFT 点数
WIN_LENGTH = 400             # 窗长 (帧长)，400点 = 25ms @16kHz
HOP_LENGTH = 320             # 帧移，320点 = 20ms @16kHz，产生约49帧/秒
FMIN = 20.0                  # 最低频率 (Hz)
FMAX = 8000.0                # 最高频率 (Hz)

# ========== 梅尔谱 ==========
N_MELS = 32                  # 梅尔频带数
LOW_FREQ_CUT = 0             # 低频裁剪起始索引 (0 表示不裁剪)
TOP_DB = 80.0                # 动态范围裁剪 (dB)，None 表示不裁剪

# ========== MFCC ==========
N_MFCC = 20                  # 保留的 MFCC 系数个数

# ========== 原始波形 ==========
RAW_TARGET_LENGTH = 8000     # 下采样后/裁切后的采样点数
RAW_FRAME_LEN = 400          # 分帧时的帧长
RAW_HOP_LEN = 160            # 分帧时的帧移

# ========== 轻量化模型 ==========
MODEL_INPUT_SHAPE = (49, 32, 1)  # 对应 log-mel: 49帧, 32梅尔频带, 1通道


# MobileNetV3 风格瓶颈参数
MODEL_WIDTH_MULTIPLIER = 0.5     # 宽度系数，控制通道数（降低参数量）
MODEL_DROPOUT = 0.3              # Dropout 比率
GRU_UNITS = 64                   # GRU 隐藏单元数

# 训练相关
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
TRAIN_DATA_DIR = "data/train"    # 训练数据路径（需自行组织）
VAL_DATA_DIR = "data/val"
MODEL_SAVE_PATH = "models/saved/audio_detector.h5"
TFLITE_MODEL_PATH = "models/saved/audio_detector.tflite"

# ========== ESC-50 数据集 ==========
ESC_50_AUDIO_DIR = "data/ESC-50-master/audio"        # 解压后的 audio 目录
ESC_50_META_PATH = "data/ESC-50-master/meta/esc50.csv"  # meta/esc50.csv
TARGET_CLASSES = {
    'dog':['dog'],
    'cat':['cat']
}

CLASS_NAMES = list(TARGET_CLASSES.keys())
NUM_CLASSES = len(CLASS_NAMES)

# 数据增强参数
AUG_SPEC_AUGMENT = True      # 时间/频率掩蔽
AUG_NOISE_SNR = [0, 15]      # 信噪比范围 (dB)
AUG_TIME_STRETCH = [0.9, 1.1] # 时间拉伸系数范围
AUG_PITCH_SHIFT = [-2, 2]     # 半音偏移范围

# 五折交叉验证
N_FOLDS = 5

# ========== 改进训练超参数 ==========
USE_KD = False                        # 是否使用知识蒸馏
KD_TEMPERATURE = 2.0                  # 蒸馏温度
KD_ALPHA = 0.7                        # 软标签损失权重
TEACHER_MODEL_PATH = "models/teacher/ast_tiny"  # 预训练教师模型目录（需自行下载）

USE_MIXUP = True                    # 是否使用频谱 Mixup
MIXUP_ALPHA = 0.4                     # Mixup Beta 分布参数

# 训练调度
WARMUP_EPOCHS = 5
MAX_LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 1e-6
LABEL_SMOOTHING = 0.0

# ========== GPU训练优化参数 ==========
USE_GPU = True                       # 是否启用GPU训练
MIXED_PRECISION = True               # 是否启用混合精度（FP16+FP32）
XLA_JIT = True                       # 是否启用XLA JIT编译加速

# 数据加载优化
NUM_WORKERS = 4                      # tf.data并行加载worker数
PREFETCH_BUFFER = 2                  # tf.data预取batch数
CACHE_DATASET = True                 # 是否缓存预处理后的数据到内存

# GPU内存增长（避免一次性占满显存）
GPU_MEMORY_GROWTH = True             # 动态增长GPU内存，而非一次性分配

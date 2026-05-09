"""
全局配置参数 - 含实验控制开关
v1.1: 关闭所有可能破坏脉冲特征的增强，提升基线稳定性
"""
# ========== 音频采集 ==========
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAMES_PER_BUFFER = 512
SEGMENT_SECONDS = 1.0
SEGMENT_SIZE = int(SAMPLE_RATE * SEGMENT_SECONDS)
QUEUE_MAX_SIZE = 100

# ========== 特征提取 ==========
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 320
FMIN = 20.0
FMAX = 8000.0
N_MELS = 32
LOW_FREQ_CUT = 0
TOP_DB = 80.0

# ========== 轻量化模型 ==========
MODEL_INPUT_SHAPE = (49, 32, 1)
MODEL_WIDTH_MULTIPLIER = 0.5
MODEL_DROPOUT = 0.3
GRU_UNITS = 64

# 训练相关
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
TRAIN_DATA_DIR = "data/train"
VAL_DATA_DIR = "data/val"
MODEL_SAVE_PATH = "models/saved/audio_detector.h5"
TFLITE_MODEL_PATH = "models/saved/audio_detector.tflite"

# ========== ESC-50 数据集 ==========
ESC_50_AUDIO_DIR = "data/ESC-50-master/audio"
ESC_50_META_PATH = "data/ESC-50-master/meta/esc50.csv"

# 目标类别（可自由修改）
TARGET_CLASSES = {
    'clock_tick': ['clock_tick']
}
CLASS_NAMES = list(TARGET_CLASSES.keys()) + ['other']
NUM_CLASSES = len(CLASS_NAMES)
OTHER_RATIO = 0.5   # 保持 target 和 other 样本数量平衡

# ========== 数据增强参数（v1.1 全部关闭） ==========
AUG_SPEC_AUGMENT = False    # 关闭频谱增强
AUG_NOISE_SNR = [30, 40]    # 极高的 SNR（几乎听不见噪声）
AUG_TIME_STRETCH = [1.0, 1.0]   # 不拉伸
AUG_PITCH_SHIFT = [0, 0]        # 不移调
AUG_RANDOM_GAIN = False         # 新增：关闭随机增益

# 交叉验证
N_FOLDS = 5

# ========== 训练超参数 ==========
USE_KD = False
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.7
TEACHER_MODEL_PATH = "models/teacher/ast_tiny"

USE_MIXUP = False           # 关闭 Mixup
MIXUP_ALPHA = 0.4

WARMUP_EPOCHS = 5
MAX_LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 1e-6
LABEL_SMOOTHING = 0.0       # 关闭标签平滑，使用硬标签

# GPU选项
USE_GPU = True
MIXED_PRECISION = True
XLA_JIT = True
NUM_WORKERS = 4
PREFETCH_BUFFER = 2
CACHE_DATASET = False
GPU_MEMORY_GROWTH = True

# ========== 自定义背景音 ==========
BACKGROUND_FILES = [
    "data/my_background.wav",
]
BACKGROUND_RATIO = 0.3   # 保留背景音混合比例，但用于 other 类构建

# ========== 开集识别方法选择 ==========
OOD_METHOD = "msp"

# 能量分数参数
ENERGY_TEMPERATURE = 1.0
ENERGY_THRESHOLD = None
ENERGY_CALIB_PATH = "config/energy_threshold.json"

# GMM参数
GMM_N_COMPONENTS = 3
GMM_MODEL_PATH = "models/saved/gmm_model.pkl"
GMM_THRESHOLD_PATH = "config/gmm_threshold.json"
GMM_CONFIDENCE_LEVEL = 0.95

# 动态阈值参数
USE_DYNAMIC_THRESHOLD = True
ENV_CALIBRATION_SECONDS = 15
THRESHOLD_STD_MULTIPLIER = 3.0
THRESHOLD_SAVE_PATH = "config/threshold.json"
FIXED_CONFIDENCE_THRESHOLD = 0.7

# 量化实验选项
QUANTIZATION_ENABLED = False
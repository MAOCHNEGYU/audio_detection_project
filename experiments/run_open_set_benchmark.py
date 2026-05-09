"""
开集识别基准测试：对比 MSP、能量、GMM 三种方法。
需要提前准备好未知类测试数据（ESC-50 剩余类别或其他）。
"""
import os, sys, json
import numpy as np
import tensorflow as tf
from utils.project_init import set_project_root
set_project_root()

import config
from models.inference import TFLiteInference
from utils.calibration import calibrate_threshold, calibrate_energy_threshold
from utils.ood_detector import GMMDetector
from utils.metrics import compute_known_unknown_auroc
from features.mel_spectrogram import MelSpectrogram

# 加载模型
infer = TFLiteInference(config.TFLITE_MODEL_PATH)
mel_ext = MelSpectrogram()

# 模拟已知/未知测试数据（此处需替换为实际加载）
# known_samples, unknown_samples 分别为已知类和未知类的特征数组 (N,49,32,1)
# y_true: 1=known, 0=unknown
# 示例省略加载过程，假设已有数据
X_known = ...
X_unknown = ...
y_true = np.array([1]*len(X_known) + [0]*len(X_unknown))

# 方法1: MSP
probs_all = np.array([infer.predict(x) for x in np.concatenate([X_known, X_unknown])])
msp_scores = np.max(probs_all, axis=1)
auroc_msp = compute_known_unknown_auroc(y_true, msp_scores)

# 方法2: 能量分数
energy_scores = -np.log(msp_scores + 1e-10)
auroc_energy = compute_known_unknown_auroc(y_true, energy_scores)

# 方法3: GMM（需提前训练好的GMM模型）
if os.path.exists(config.GMM_MODEL_PATH):
    gmm = GMMDetector(config.GMM_MODEL_PATH)
    # 需要特征提取模型，此处假设已实现
    # 仅示例，省略具体特征提取过程
    auroc_gmm = 0.0
else:
    auroc_gmm = None

print(f"AUROC: MSP={auroc_msp:.4f}, Energy={auroc_energy:.4f}, GMM={auroc_gmm}")
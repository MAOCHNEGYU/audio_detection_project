"""
消融实验：测试不同数据增强组合的影响。
遍历 Mixup、SpecAugment、时间拉伸、音高偏移的开关组合。
"""
import os, sys, itertools, json
import numpy as np
import tensorflow as tf
from utils.project_init import set_project_root
set_project_root()

import config
from data.dataset import ESC50Dataset
from models.lightweight_model import create_tiny_model
from models.train import train_fold

# 实验配置覆盖
base_config = {
    'USE_MIXUP': True,
    'AUG_SPEC_AUGMENT': True,
    'AUG_TIME_STRETCH': [0.9, 1.1],
    'AUG_PITCH_SHIFT': [-2, 2],
    'EPOCHS': 20,       # 快速实验
    'N_FOLDS': 3
}

enhancements = ['MIXUP', 'SPECAUG', 'TIMESTRETCH', 'PITCHSHIFT']
results = {}

for r in range(len(enhancements)+1):
    for combo in itertools.combinations(enhancements, r):
        combo_name = '_'.join(combo) if combo else 'BASELINE'
        print(f"\n{'='*60}\n消融实验: {combo_name}\n{'='*60}")

        # 根据 combo 覆盖配置
        config.USE_MIXUP = ('MIXUP' in combo)
        config.AUG_SPEC_AUGMENT = ('SPECAUG' in combo)
        config.AUG_TIME_STRETCH = [0.9,1.1] if 'TIMESTRETCH' in combo else [1.0,1.0]
        config.AUG_PITCH_SHIFT = [-2,2] if 'PITCHSHIFT' in combo else [0,0]
        config.EPOCHS = 20
        config.N_FOLDS = 3

        fold_acc = []
        for fold in range(config.N_FOLDS):
            best_val_acc, f1, _ = train_fold(fold, None)
            fold_acc.append(best_val_acc)
        mean_acc = np.mean(fold_acc)
        results[combo_name] = mean_acc
        print(f"组合 {combo_name} 平均准确率: {mean_acc:.4f}")

print("\n最终消融结果:")
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")

with open("experiments/ablation_results.json", 'w') as f:
    json.dump(results, f, indent=2)
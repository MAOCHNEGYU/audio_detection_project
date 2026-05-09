"""
评估指标工具：计算 F1、混淆矩阵、AUROC 等。
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

def compute_metrics(y_true, y_pred, class_names=None):
    """
    返回包含准确率、加权F1、混淆矩阵等的字典。
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    result = {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1_weighted': f,
        'confusion_matrix': cm
    }
    if class_names:
        result['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
    return result

def compute_known_unknown_auroc(y_true, y_score):
    """
    y_true: 0 = unknown, 1 = known
    y_score: 已知类的置信度分数（越大越可能已知）
    返回 AUROC
    """
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.0
"""
可视化工具：绘制混淆矩阵、训练曲线等。
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_training_curves(history, save_path=None):
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(history['loss'], label='train')
    axs[1].plot(history['val_accuracy'], label='val')
    axs[0].set_title('Loss'); axs[0].legend()
    axs[1].set_title('Accuracy'); axs[1].legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
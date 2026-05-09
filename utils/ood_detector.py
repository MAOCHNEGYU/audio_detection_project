"""
GMM 开集检测器：基于马氏距离判断样本是否属于已知类。
"""
import numpy as np
from scipy.stats import chi2
import joblib
import config


class GMMDetector:
    """
    加载训练好的 GMM 模型，计算马氏距离并判断是否已知。
    """
    def __init__(self, model_path):
        self.gmm = joblib.load(model_path)
        self.dim = self.gmm.means_.shape[1]
        self.threshold = np.sqrt(chi2.ppf(config.GMM_CONFIDENCE_LEVEL, self.dim))

    def score(self, features):
        """
        features: 1D 或 2D 特征向量
        返回: (is_known, min_mahalanobis_distance)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        dists = []
        for mean, cov in zip(self.gmm.means_, self.gmm.covariances_):
            diff = features - mean
            try:
                inv_cov = np.linalg.inv(cov)
                dist = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
            except np.linalg.LinAlgError:
                # 协方差奇异时使用对角元素
                inv_cov = np.diag(1.0 / np.diag(cov))
                dist = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
            dists.append(dist)
        min_dist = np.min(dists, axis=0)[0]
        is_known = min_dist < self.threshold
        return is_known, float(min_dist)


def train_and_save_gmm(features, save_path):
    """
    使用已知类特征训练 GMM 并保存。
    features: (N, D)
    """
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=config.GMM_N_COMPONENTS,
                          covariance_type='full',
                          random_state=42)
    gmm.fit(features)
    joblib.dump(gmm, save_path)
    return gmm
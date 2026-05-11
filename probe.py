"""
probe.py — Scaler -> PCA(pooled) -> Scaler(geo) -> Catboost
Includes SHAP analysis method for interpretation.
"""
from __future__ import annotations

import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier


class HallucinationProbe(nn.Module):
    """Sklearn-based probe matching the required public API."""
    def __init__(self, geo_dim: int = 6, n_components: int = 10) -> None:
        super().__init__()
        self.scaler_p = StandardScaler()
        self.scaler_g = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        
        # class_weight='balanced' automatically handles class imbalance
        self.clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.02,
            depth=3,
            l2_leaf_reg=10.0,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False,
            task_type='CPU'
        )
        self._threshold: float = 0.5
        self._geo_dim: int = geo_dim
        self._n_components: int = n_components
        self._is_fitted: bool = False

    def _process(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        p_dim = X.shape[1] - self._geo_dim
        X_p, X_g = X[:, :p_dim], X[:, p_dim:]
        
        X_p_sc = self.scaler_p.fit_transform(X_p) if fit else self.scaler_p.transform(X_p)
        X_p_pca = self.pca.fit_transform(X_p_sc) if fit else self.pca.transform(X_p_sc)
        
        if self._geo_dim > 0:
            X_g_sc = self.scaler_g.fit_transform(X_g) if fit else self.scaler_g.transform(X_g)
            return np.hstack([X_p_pca, X_g_sc])
        return X_p_pca

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_final = self._process(X, fit=True)
        self.clf.fit(X_final, y)
        self._is_fitted = True
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X_final = self._process(X, fit=False)
        prob_pos = self.clf.predict_proba(X_final)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def plot_shap_analysis(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None, 
        feature_names: list[str] = None, 
        output_dir: str = "./outputs"
    ) -> None:
        """
        Computes SHAP values and saves global importance plots to output_dir.
        """
        if not self._is_fitted:
            raise RuntimeError("Probe must be fitted before SHAP analysis.")

        os.makedirs(output_dir, exist_ok=True)
        
        X_processed = self._process(X, fit=False)
        
        if feature_names is None:
            names = [f"PCA_{i}" for i in range(self._n_components)]
            names += [f"Geo_{i}" for i in range(self._geo_dim)]
        else:
            names = feature_names

        explainer = shap.TreeExplainer(self.clf)

        shap_values = explainer.shap_values(X_processed)
        
        shap_exp = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_processed,
            feature_names=names
        )

        # Plot 1: Global Feature Importance (Bar)
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_exp, max_display=20) # Show top 20 features
        plt.tight_layout()
        bar_path = os.path.join(output_dir, "shap_bar_importance.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP Bar Plot to {bar_path}")
        plt.close()
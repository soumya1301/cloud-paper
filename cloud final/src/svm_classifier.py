"""
SVM Classifier Module
Implements the Support Vector Machine (SVM) classifier with RBF kernel
trained on the 5 latent features extracted by the Autoencoder, matching
Section 4 & 6.3 and Table 5 of the paper (Alelyani et al., 2026).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class VMSVMClassifier:
    """
    SVM Classifier with RBF kernel operating on 5 latent features.
    Classifies VMs into:
      - Class 1: Balanced / Low CPU
      - Class 2: CPU-Intensive
      - Class 3: Moderate CPU / Low Memory
    """
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: Any = "scale",
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=random_state
        )
        self.best_params = {}
        self.is_fitted = False

    def fit_with_cv(
        self,
        X_latent_train: np.ndarray,
        y_train: np.ndarray,
        c_range: List[float] = [0.1, 1.0, 10.0, 100.0],
        gamma_range: List[Any] = ["scale", "auto", 0.01, 0.1, 1.0]
    ) -> "VMSVMClassifier":
        """
        Cross-validates C and gamma hyperparameters on latent features,
        then fits the optimal SVM model.
        """
        print(f"[*] Running {self.cv_folds}-fold Cross-Validation for SVM (RBF kernel)...")
        param_grid = {"C": c_range, "gamma": gamma_range}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            SVC(kernel=self.kernel, probability=True, random_state=self.random_state),
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid_search.fit(X_latent_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_fitted = True

        print(f"[+] Optimal SVM Hyperparameters found: C={self.best_params['C']}, gamma={self.best_params['gamma']}")
        return self

    def fit(self, X_latent: np.ndarray, y: np.ndarray) -> "VMSVMClassifier":
        """Directly fits SVM model without CV."""
        self.model.fit(X_latent, y)
        self.is_fitted = True
        return self

    def predict(self, X_latent: np.ndarray) -> np.ndarray:
        """Predicts VM class (1, 2, or 3) from 5 latent features."""
        if not self.is_fitted:
            raise RuntimeError("VMSVMClassifier must be fitted before predict.")
        return self.model.predict(X_latent)

    def predict_proba(self, X_latent: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("VMSVMClassifier must be fitted before predict.")
        return self.model.predict_proba(X_latent)

    def evaluate(self, X_latent_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates classifier performance matching Table 5 of the paper:
        Accuracy, Precision, Recall/Sensitivity, F1-Score per class and macro/weighted averages.
        """
        y_pred = self.predict(X_latent_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        classes = np.unique(np.concatenate([y_test, y_pred]))
        class_table = []
        for i, c in enumerate(classes):
            class_table.append({
                "Class Type": f"Class {c}",
                "precision": round(float(prec_per_class[i]), 3),
                "sensitivity (recall)": round(float(rec_per_class[i]), 3),
                "F1-score": round(float(f1_per_class[i]), 3)
            })

        cm = confusion_matrix(y_test, y_pred)

        results = {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_weighted),
            "class_metrics_df": pd.DataFrame(class_table),
            "confusion_matrix": cm,
            "classification_report_str": classification_report(y_test, y_pred, digits=3)
        }
        return results

    def save(self, filepath: str = "models/svm.pkl"):
        """Saves SVM classifier to pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Saved SVM model to '{filepath}'")

    @classmethod
    def load(cls, filepath: str = "models/svm.pkl") -> "VMSVMClassifier":
        """Loads SVM classifier from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def plot_confusion_matrix(
        self,
        X_latent_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str = "results/figures/svm_confusion_matrix.png"
    ):
        """Plots confusion matrix heatmap."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        y_pred = self.predict(X_latent_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Class 1 (Balanced)", "Class 2 (CPU-Int)", "Class 3 (Mod-CPU)"],
            yticklabels=["Class 1 (Balanced)", "Class 2 (CPU-Int)", "Class 3 (Mod-CPU)"]
        )
        plt.title("SVM Confusion Matrix (5 Latent Features)", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved SVM confusion matrix to '{save_path}'")

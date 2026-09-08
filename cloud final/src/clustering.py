"""
K-Means VM Clustering Module
Implements K-Means clustering with K=3 as described in Section 4 & 6.3 of the paper.
Analyzes cluster resource characteristics and maps numerical cluster IDs to semantic VM types.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class VMClusterer:
    """
    K-Means clustering engine for Virtual Machine classification.
    Clusters VMs into 3 categories:
      - Class 1: Balanced / Low CPU
      - Class 2: CPU-Intensive (High CPU demand, limited to 25% per PM)
      - Class 3: Moderate CPU / Low Memory
    """
    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        max_iter: int = 300,
        init: str = "k-means++"
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.init = init
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init=init,
            n_init=10
        )
        self.cluster_mapping = {} # maps raw cluster id -> semantic paper cluster (1, 2, 3)
        self.is_fitted = False

    def fit(self, X: np.ndarray, df_original: pd.DataFrame = None) -> "VMClusterer":
        """
        Fits K-Means on feature matrix X.
        Analyzes cluster CPU and memory profiles to map raw cluster IDs to
        Paper Classes:
          - Class 2: Cluster with highest mean CPU utilization (CPU-Intensive)
          - Class 1: Cluster with lowest mean CPU utilization (Balanced / Low CPU)
          - Class 3: Cluster with intermediate CPU utilization (Moderate CPU / Low Memory)
        """
        raw_labels = self.kmeans.fit_predict(X)
        self.is_fitted = True

        # Determine semantic cluster mapping based on actual CPU and memory usage
        if df_original is not None and "cpu_avg" in df_original.columns:
            cpu_means = {}
            for k in range(self.n_clusters):
                mask = (raw_labels == k)
                cpu_means[k] = df_original.loc[mask, "cpu_avg"].mean()

            # Sort raw cluster IDs by mean CPU usage ascending
            sorted_by_cpu = sorted(cpu_means.keys(), key=lambda k: cpu_means[k])
            
            # Lowest CPU -> Class 1 (Balanced/Low CPU)
            # Highest CPU -> Class 2 (CPU-Intensive)
            # Intermediate CPU -> Class 3 (Moderate CPU/Low Memory)
            self.cluster_mapping = {
                sorted_by_cpu[0]: 1, # Class 1
                sorted_by_cpu[2]: 2, # Class 2 (CPU-Intensive)
                sorted_by_cpu[1]: 3  # Class 3
            }
        else:
            # Default identity mapping if no raw DataFrame provided
            self.cluster_mapping = {k: k + 1 for k in range(self.n_clusters)}

        print(f"[*] K-Means fitted with {self.n_clusters} clusters.")
        print(f"    Raw-to-Semantic Cluster Mapping: {self.cluster_mapping}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts semantic paper cluster IDs (1, 2, or 3) for feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("VMClusterer must be fitted before predict.")
        raw_labels = self.kmeans.predict(X)
        mapped_labels = np.array([self.cluster_mapping.get(l, l + 1) for l in raw_labels])
        return mapped_labels

    def get_cluster_stats(self, X: np.ndarray, df_original: pd.DataFrame) -> pd.DataFrame:
        """Computes mean and standard deviation for CPU and Memory across clusters."""
        labels = self.predict(X)
        df = df_original.copy()
        df["cluster"] = labels

        stats = []
        for c in [1, 2, 3]:
            df_c = df[df["cluster"] == c]
            stats.append({
                "cluster": c,
                "name": "Balanced/Low-CPU" if c == 1 else ("CPU-Intensive" if c == 2 else "Moderate-CPU/Low-Mem"),
                "count": len(df_c),
                "proportion": len(df_c) / len(df),
                "mean_cpu": df_c["cpu_avg"].mean(),
                "std_cpu": df_c["cpu_avg"].std(),
                "mean_mem": df_c["mem_avg"].mean(),
                "std_mem": df_c["mem_avg"].std(),
                "max_mem": df_c["mem_max"].mean()
            })
        return pd.DataFrame(stats)

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Calculates clustering quality metrics."""
        raw_labels = self.kmeans.predict(X)
        sil_score = silhouette_score(X, raw_labels) if len(X) > self.n_clusters else 0.0
        inertia = float(self.kmeans.inertia_)
        return {
            "silhouette_score": float(sil_score),
            "inertia": inertia,
            "n_clusters": self.n_clusters
        }

    def save(self, filepath: str = "models/kmeans.pkl"):
        """Saves fitted clusterer to pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Saved K-Means model to '{filepath}'")

    @classmethod
    def load(cls, filepath: str = "models/kmeans.pkl") -> "VMClusterer":
        """Loads fitted clusterer from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def plot_clusters(
        self,
        X: np.ndarray,
        df_original: pd.DataFrame,
        save_path: str = "results/figures/kmeans_clusters.png"
    ):
        """Generates visualizations showing cluster distribution and resource profiles."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        labels = self.predict(X)
        df = df_original.copy()
        df["cluster"] = labels

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Cluster Counts
        palette = {1: "#2b5c8f", 2: "#d95f02", 3: "#7570b3"}
        sns.countplot(data=df, x="cluster", hue="cluster", ax=axes[0], palette=palette, legend=False)
        axes[0].set_title("VM Count by Cluster (K=3)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Cluster ID", fontsize=11)
        axes[0].set_ylabel("Number of VMs", fontsize=11)
        for p in axes[0].patches:
            height = p.get_height()
            if height > 0:
                axes[0].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                 ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        # 2. CPU vs Memory Scatter / Centroids
        sns.scatterplot(
            data=df,
            x="cpu_avg",
            y="mem_avg",
            hue="cluster",
            palette=palette,
            alpha=0.6,
            ax=axes[1]
        )
        axes[1].set_title("CPU vs Memory Distribution by Cluster", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Mean CPU Utilization", fontsize=11)
        axes[1].set_ylabel("Mean Memory Utilization", fontsize=11)

        # 3. Mean Resource Utilization Bar Chart (Matching Paper Fig 6)
        stats = self.get_cluster_stats(X, df)
        x_pos = np.arange(3)
        width = 0.35
        axes[2].bar(x_pos - width/2, stats["mean_cpu"], width, label="Mean CPU", color="#d95f02")
        axes[2].bar(x_pos + width/2, stats["mean_mem"], width, label="Mean Memory", color="#2b5c8f")
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(["Cluster 1\n(Balanced)", "Cluster 2\n(CPU-Intensive)", "Cluster 3\n(Moderate/Low-Mem)"])
        axes[2].set_title("Mean Resource Utilization (Paper Fig. 6)", fontsize=13, fontweight="bold")
        axes[2].set_ylabel("Mean Utilization Ratio", fontsize=11)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved K-Means visualization to '{save_path}'")

"""
Notebook Generator Script
Generates the 5 structured Jupyter notebooks in notebooks/
"""

import os
import nbformat as nbf

def create_notebooks():
    os.makedirs("notebooks", exist_ok=True)

    # 1. 01_dataset_analysis.ipynb
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell("# 01. Google Cluster Workload Trace - Dataset Analysis\n\nThis notebook loads and analyzes the Google Cluster Trace dataset (Alelyani et al., 2026), inspecting the 16 extracted features across the 5-minute observation window."),
        nbf.v4.new_code_cell("""import os, sys, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "..")
from src.data_preprocessing import DataPreprocessor, FEATURE_NAMES

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# Ensure processed data is available
preprocessor = DataPreprocessor(config)
df_train = pd.read_csv("../data/processed/train_features.csv")
df_sched = pd.read_csv("../data/processed/scheduling_workload.csv")

print(f"ML Training Set VMs: {len(df_train)}")
print(f"Scheduling Workload VMs: {len(df_sched)}")
df_train[FEATURE_NAMES].describe().T
"""),
        nbf.v4.new_code_cell("""# Plot distributions of 16 VM features
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(FEATURE_NAMES):
    sns.histplot(df_train[col], bins=25, kde=True, ax=axes[i], color="#025e8d")
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")

plt.tight_layout()
plt.show()
""")
    ]
    with open("notebooks/01_dataset_analysis.ipynb", "w") as f:
        nbf.write(nb1, f)

    # 2. 02_kmeans.ipynb
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell("# 02. K-Means Clustering (K=3)\n\nClustering VMs based on the 16 resource utilization features into 3 classes:\n- **Class 1**: Balanced / Low CPU\n- **Class 2**: CPU-Intensive\n- **Class 3**: Moderate CPU / Low Memory"),
        nbf.v4.new_code_cell("""import os, sys, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "..")
from src.clustering import VMClusterer

X_train = np.load("../data/processed/X_train.npy")
df_train = pd.read_csv("../data/processed/train_features.csv")

clusterer = VMClusterer(n_clusters=3, random_state=42)
clusterer.fit(X_train, df_original=df_train)
stats = clusterer.get_cluster_stats(X_train, df_train)
stats
"""),
        nbf.v4.new_code_cell("""# Visualize Clusters (Paper Fig 6)
clusterer.plot_clusters(X_train, df_train, save_path="../results/figures/kmeans_clusters.png")
from IPython.display import Image
Image(filename="../results/figures/kmeans_clusters.png")
""")
    ]
    with open("notebooks/02_kmeans.ipynb", "w") as f:
        nbf.write(nb2, f)

    # 3. 03_autoencoder.ipynb
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell("# 03. Autoencoder (16 -> 12 -> 5 -> 12 -> 16)\n\nTraining the PyTorch Autoencoder to compress 16 VM features into 5 latent features."),
        nbf.v4.new_code_cell("""import os, sys, yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from src.autoencoder import VMAutoencoder

X_train = np.load("../data/processed/X_train.npy")
X_val = np.load("../data/processed/X_val.npy")
X_test = np.load("../data/processed/X_test.npy")

ae = VMAutoencoder(input_dim=16, latent_dim=5, epochs=500, batch_size=64)
ae.fit(X_train, X_val=X_val, verbose=True)

metrics = ae.evaluate_reconstruction(X_test)
print(f"Reconstruction MSE: {metrics['test_mse']:.6f}")
print(f"Reconstruction Accuracy: {metrics['reconstruction_accuracy']:.2f}%")
"""),
        nbf.v4.new_code_cell("""# Loss Curve (Paper Fig. 5a)
ae.plot_loss(save_path="../results/figures/fig5_autoencoder_loss.png")
from IPython.display import Image
Image(filename="../results/figures/fig5_autoencoder_loss.png")
""")
    ]
    with open("notebooks/03_autoencoder.ipynb", "w") as f:
        nbf.write(nb3, f)

    # 4. 04_svm.ipynb
    nb4 = nbf.v4.new_notebook()
    nb4.cells = [
        nbf.v4.new_markdown_cell("# 04. SVM Classifier on 5 Latent Features\n\nTraining Support Vector Machine with RBF kernel on Autoencoder latent representations matching Table 5."),
        nbf.v4.new_code_cell("""import os, sys, yaml
import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier
from src.clustering import VMClusterer

ae = VMAutoencoder.load("../models/autoencoder.pth")
clusterer = VMClusterer.load("../models/kmeans.pkl")

X_train = np.load("../data/processed/X_train.npy")
X_test = np.load("../data/processed/X_test.npy")

y_train = clusterer.predict(X_train)
y_test = clusterer.predict(X_test)

X_latent_train = ae.extract_latent(X_train)
X_latent_test = ae.extract_latent(X_test)

svm = VMSVMClassifier(kernel="rbf")
svm.fit_with_cv(X_latent_train, y_train)

eval_res = svm.evaluate(X_latent_test, y_test)
print(eval_res["class_metrics_df"].to_markdown(index=False))
print(f"Accuracy: {eval_res['accuracy']:.3f}")
"""),
        nbf.v4.new_code_cell("""# Confusion Matrix
svm.plot_confusion_matrix(X_latent_test, y_test, save_path="../results/figures/svm_confusion_matrix.png")
from IPython.display import Image
Image(filename="../results/figures/svm_confusion_matrix.png")
""")
    ]
    with open("notebooks/04_svm.ipynb", "w") as f:
        nbf.write(nb4, f)

    # 5. 05_results.ipynb
    nb5 = nbf.v4.new_notebook()
    nb5.cells = [
        nbf.v4.new_markdown_cell("# 05. End-to-End Simulation & Benchmark Comparison\n\nReproducing Table 6 and all experimental comparison figures across 798 PMs and 7,644 VMs."),
        nbf.v4.new_code_cell("""import os, sys, yaml
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from experiments.run_comparison import main as run_comparison_main

# Run comparative evaluation
run_comparison_main()
"""),
        nbf.v4.new_code_cell("""# View Table 6
df_table6 = pd.read_csv("../results/tables/comparison_table.csv")
df_table6
"""),
        nbf.v4.new_code_cell("""# View Comparison Metrics Plots
from IPython.display import Image
Image(filename="../results/figures/fig_comparison_metrics.png")
""")
    ]
    with open("notebooks/05_results.ipynb", "w") as f:
        nbf.write(nb5, f)

    print("[+] Created all 5 Jupyter Notebooks in notebooks/")

if __name__ == "__main__":
    create_notebooks()

"""
ML Training Pipeline
Executes the end-to-end Machine Learning training workflow from Section 4 & 6.3 of the paper:
1. Load & clean Google trace dataset
2. Scale 16 features
3. Run K-Means clustering (K=3) -> assign labels
4. Train PyTorch Autoencoder (16 -> 12 -> 5 -> 12 -> 16) -> extract 5 latent features
5. Train SVM Classifier (RBF kernel) on 5 latent features with cross-validation
6. Output Table 5 classification metrics and save models & plots
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import DataPreprocessor
from src.clustering import VMClusterer
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("random_seed", 42)
    set_seeds(seed)

    print("=" * 70)
    print(" STEP 1: PREPROCESSING & DATA SPLIT")
    print("=" * 70)
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.process_and_save()
    
    df_train = data_dict["df_train"]
    df_val = data_dict["df_val"]
    df_test = data_dict["df_test"]
    X_train = data_dict["X_train"]
    X_val = data_dict["X_val"]
    X_test = data_dict["X_test"]

    print("\n" + "=" * 70)
    print(" STEP 2: K-MEANS CLUSTERING (K=3)")
    print("=" * 70)
    kmeans_cfg = config.get("ml", {}).get("kmeans", {})
    clusterer = VMClusterer(
        n_clusters=kmeans_cfg.get("n_clusters", 3),
        random_state=kmeans_cfg.get("random_state", seed),
        max_iter=kmeans_cfg.get("max_iter", 300),
        init=kmeans_cfg.get("init", "k-means++")
    )
    clusterer.fit(X_train, df_original=df_train)
    
    # Predict clusters (labels: 1, 2, 3)
    y_train = clusterer.predict(X_train)
    y_val = clusterer.predict(X_val)
    y_test = clusterer.predict(X_test)

    cluster_stats = clusterer.get_cluster_stats(X_train, df_train)
    print("\n[+] K-Means Cluster Statistics (Training Set):")
    print(cluster_stats.to_string(index=False))

    # Save model and plot
    clusterer.save("models/kmeans.pkl")
    clusterer.plot_clusters(X_train, df_train, save_path="results/figures/fig6_cluster_utilization.png")

    print("\n" + "=" * 70)
    print(" STEP 3: AUTOENCODER (16 -> 12 -> 5 -> 12 -> 16)")
    print("=" * 70)
    ae_cfg = config.get("ml", {}).get("autoencoder", {})
    autoencoder = VMAutoencoder(
        input_dim=ae_cfg.get("input_dim", 16),
        hidden_dim1=ae_cfg.get("hidden_dim1", 12),
        latent_dim=ae_cfg.get("latent_dim", 5),
        hidden_dim2=ae_cfg.get("hidden_dim2", 12),
        output_dim=ae_cfg.get("output_dim", 16),
        learning_rate=ae_cfg.get("learning_rate", 0.001),
        epochs=ae_cfg.get("epochs", 500),
        batch_size=ae_cfg.get("batch_size", 64),
        weight_decay=ae_cfg.get("weight_decay", 1e-5)
    )
    autoencoder.fit(X_train, X_val=X_val, verbose=True)
    
    # Evaluate reconstruction
    recon_eval = autoencoder.evaluate_reconstruction(X_test)
    print(f"\n[+] Autoencoder Test MSE: {recon_eval['test_mse']:.6f}")
    print(f"[+] Autoencoder Reconstruction Accuracy: {recon_eval['reconstruction_accuracy']:.2f}%")

    # Save model and loss plot
    autoencoder.save("models/autoencoder.pth")
    autoencoder.plot_loss("results/figures/fig5_autoencoder_loss.png")

    # Extract 5 latent features
    X_latent_train = autoencoder.extract_latent(X_train)
    X_latent_val = autoencoder.extract_latent(X_val)
    X_latent_test = autoencoder.extract_latent(X_test)

    print("\n" + "=" * 70)
    print(" STEP 4: SVM CLASSIFIER (RBF Kernel on 5 Latent Features)")
    print("=" * 70)
    svm_cfg = config.get("ml", {}).get("svm", {})
    svm_classifier = VMSVMClassifier(
        kernel=svm_cfg.get("kernel", "rbf"),
        cv_folds=svm_cfg.get("cv_folds", 5),
        random_state=seed
    )
    svm_classifier.fit_with_cv(
        X_latent_train,
        y_train,
        c_range=svm_cfg.get("c_range", [0.1, 1.0, 10.0, 100.0]),
        gamma_range=svm_cfg.get("gamma_range", ["scale", "auto", 0.01, 0.1, 1.0])
    )

    # Evaluate SVM on Test Set
    svm_eval = svm_classifier.evaluate(X_latent_test, y_test)
    print("\n[+] Table 5 Reproduction: Classification Performance on 5 Latent Features:")
    print(svm_eval["class_metrics_df"].to_string(index=False))
    print(f"\n    - Accuracy: {svm_eval['accuracy']:.3f} (Paper: 0.769 / 77%)")
    print(f"    - Macro Precision: {svm_eval['precision_macro']:.3f} (Paper: 0.79 / 79%)")
    print(f"    - Macro Sensitivity/Recall: {svm_eval['recall_macro']:.3f} (Paper: 0.80 / 80%)")
    print(f"    - Macro F1-Score: {svm_eval['f1_macro']:.3f} (Paper: 0.79 / 79%)")

    # Save Table 5 to markdown
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/table5_classification_results.md", "w") as f:
        f.write("# Table 5: Classification Performance using 5 Latent Features\n\n")
        f.write(svm_eval["class_metrics_df"].to_markdown(index=False))
        f.write(f"\n\n**Accuracy**: {svm_eval['accuracy']:.3f}\n")
        f.write(f"**Macro Precision**: {svm_eval['precision_macro']:.3f}\n")
        f.write(f"**Macro Sensitivity (Recall)**: {svm_eval['recall_macro']:.3f}\n")
        f.write(f"**Macro F1-Score**: {svm_eval['f1_macro']:.3f}\n")

    # Save SVM model and confusion matrix plot
    svm_classifier.save("models/svm.pkl")
    svm_classifier.plot_confusion_matrix(X_latent_test, y_test, save_path="results/figures/svm_confusion_matrix.png")

    print("\n" + "=" * 70)
    print(" [*] ML TRAINING PIPELINE COMPLETE SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()

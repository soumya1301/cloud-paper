"""
Main CLI Entrypoint
Command-line interface to execute the full pipeline for reproducing:
"A Machine Learning Technique for Optimizing Virtual Machine Placement in Data-Centres"
(Alelyani, Datta, Hassan, Journal of Cloud Computing, 2026).
"""

import os
import sys
import argparse
import yaml

from data.download_subset import main as download_main
from src.data_preprocessing import DataPreprocessor
from experiments.train_ml import main as train_ml_main
from experiments.run_scheduler import run_proposed_scheduler_experiment
from experiments.run_comparison import main as compare_main

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="ML-Based VM Placement in Data-Centres (Alelyani et al., 2026 Reproduction)"
    )
    parser.add_argument(
        "action",
        choices=["download", "preprocess", "train_ml", "simulate", "compare", "all"],
        help="Action to execute: 'download', 'preprocess', 'train_ml', 'simulate', 'compare', or 'all'"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (faster, smaller subset)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dev:
        config["simulation"]["dev_mode"] = True
        print("[!] Development mode enabled (smaller PM and VM count).")

    if args.action in ["download", "all"]:
        print("\n" + "=" * 80)
        print(">>> STEP 1: DOWNLOADING / EXTRACTING GOOGLE CLUSTER TRACE SUBSET")
        print("=" * 80)
        download_main([])

    if args.action in ["preprocess", "all"]:
        print("\n" + "=" * 80)
        print(">>> STEP 2: PREPROCESSING DATA & EXTRACTING 16 FEATURES")
        print("=" * 80)
        preprocessor = DataPreprocessor(config)
        preprocessor.process_and_save()

    if args.action in ["train_ml", "all"]:
        print("\n" + "=" * 80)
        print(">>> STEP 3: TRAINING ML PIPELINE (K-MEANS -> AUTOENCODER -> SVM)")
        print("=" * 80)
        train_ml_main()

    if args.action in ["simulate", "all"]:
        print("\n" + "=" * 80)
        print(">>> STEP 4: RUNNING PROPOSED CLOUD SCHEDULER SIMULATION")
        print("=" * 80)
        run_proposed_scheduler_experiment(config)

    if args.action in ["compare", "all"]:
        print("\n" + "=" * 80)
        print(">>> STEP 5: COMPARATIVE BENCHMARK EVALUATION (TABLE 6)")
        print("=" * 80)
        compare_main()

    print("\n" + "=" * 80)
    print(f"[+] Task '{args.action}' completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()

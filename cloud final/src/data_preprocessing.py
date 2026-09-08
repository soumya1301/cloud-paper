"""
Data Preprocessing Pipeline
Handles loading raw Google trace tables, filtering 5-minute observation windows,
cleaning invalid records, handling missing values, scaling, and splitting datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# The 16 features identified from Section 6.2 of the paper
FEATURE_NAMES: List[str] = [
    "cpu_max",
    "cpu_min",
    "cpu_avg",
    "cpu_std",
    "cpu_p95",
    "cpi",
    "mai",
    "mem_max",
    "mem_min",
    "mem_avg",
    "mem_assigned",
    "mem_page_cache",
    "mem_canonical",
    "start_time",
    "end_time",
    "duration"
]

class DataPreprocessor:
    """
    Preprocessor for Google Cluster Workload Trace.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_dir = config.get("dataset", {}).get("raw_dir", "data/raw")
        self.processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
        self.obs_window = config.get("dataset", {}).get("observation_window_minutes", 5)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads machine_events, task_events, and task_usage tables."""
        mach_path = os.path.join(self.raw_dir, "machine_events.csv")
        task_path = os.path.join(self.raw_dir, "task_events.csv")
        usage_path = os.path.join(self.raw_dir, "task_usage.csv")

        if not all(os.path.exists(p) for p in [mach_path, task_path, usage_path]):
            raise FileNotFoundError(
                f"Raw data files not found in {self.raw_dir}. "
                "Please run data/download_subset.py first."
            )

        df_mach = pd.read_csv(mach_path)
        df_tasks = pd.read_csv(task_path)
        df_usage = pd.read_csv(usage_path)

        return df_mach, df_tasks, df_usage

    def clean_data(self, df_usage: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans trace data: removes NaNs, handles missing values,
        and ensures positive non-zero bounds for resource metrics.
        """
        df = df_usage.copy()
        
        # Fill numerical NaNs with column median or forward-fill
        for col in FEATURE_NAMES:
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
                # Clip negative or impossible values
                if "cpu" in col or "mem" in col:
                    df[col] = np.clip(df[col], 0.0, 1.0)
                elif col in ["cpi", "mai", "duration"]:
                    df[col] = np.clip(df[col], 0.0, None)

        # Drop any remaining duplicates
        df = df.drop_duplicates(subset=["vm_id"])
        return df

    def split_ml_dataset(
        self, df_ml: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the 2,293 ML VM dataset into Train (70%), Validation (10%), and Test (20%)
        as specified in Section 6.3 of the paper.
        """
        seed = self.config.get("random_seed", 42)
        n = len(df_ml)
        indices = np.random.RandomState(seed).permutation(n)
        
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        df_train = df_ml.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df_ml.iloc[val_idx].copy().reset_index(drop=True)
        df_test = df_ml.iloc[test_idx].copy().reset_index(drop=True)

        return df_train, df_val, df_test

    def fit_transform_features(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scales the 16 features using MinMaxScaler fitted on the training split.
        """
        X_train = df_train[FEATURE_NAMES].values
        X_val = df_val[FEATURE_NAMES].values
        X_test = df_test[FEATURE_NAMES].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame of features using the already-fitted scaler."""
        X = df[FEATURE_NAMES].values
        return self.scaler.transform(X)

    def process_and_save(self) -> Dict[str, Any]:
        """
        Full preprocessing pipeline: loads raw data, splits ML and Scheduling sets,
        scales features, and saves processed CSVs.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        df_mach, df_tasks, df_usage = self.load_raw_data()
        df_clean = self.clean_data(df_usage)

        # Separate ML dataset and Scheduling dataset
        df_ml = df_clean[df_clean["is_ml_set"] == True].copy().reset_index(drop=True)
        df_sched = df_clean[df_clean["is_ml_set"] == False].copy().reset_index(drop=True)

        # Split ML dataset: 70% Train, 10% Val, 20% Test
        df_train, df_val, df_test = self.split_ml_dataset(df_ml)

        # Scale features
        X_train, X_val, X_test = self.fit_transform_features(df_train, df_val, df_test)
        X_sched = self.transform_features(df_sched)

        # Save processed files
        df_train.to_csv(os.path.join(self.processed_dir, "train_features.csv"), index=False)
        df_val.to_csv(os.path.join(self.processed_dir, "val_features.csv"), index=False)
        df_test.to_csv(os.path.join(self.processed_dir, "test_features.csv"), index=False)
        df_sched.to_csv(os.path.join(self.processed_dir, "scheduling_workload.csv"), index=False)
        df_mach.to_csv(os.path.join(self.processed_dir, "physical_machines.csv"), index=False)

        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "X_sched.npy"), X_sched)

        print(f"[*] Preprocessing complete. Saved files in '{self.processed_dir}':")
        print(f"    - Train set: {len(df_train)} VMs, Val set: {len(df_val)} VMs, Test set: {len(df_test)} VMs")
        print(f"    - Scheduling set: {len(df_sched)} VMs across {len(df_mach)} PMs")

        return {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "df_sched": df_sched,
            "df_mach": df_mach,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_sched": X_sched,
            "scaler": self.scaler
        }

if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    preprocessor = DataPreprocessor(cfg)
    preprocessor.process_and_save()

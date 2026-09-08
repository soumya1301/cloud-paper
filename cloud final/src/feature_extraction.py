"""
Feature Extraction Module
Extracts the 16 VM-related features from the 5-minute observation window
as defined in the research paper (Alelyani et al., 2026).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

FEATURE_NAMES = [
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

class FeatureExtractor:
    """
    Extracts 16 VM features from 5-minute workload observation traces.
    """
    def __init__(self, observation_window_minutes: float = 5.0):
        self.obs_window_seconds = observation_window_minutes * 60.0
        self.feature_names = FEATURE_NAMES

    def extract_from_time_series(
        self,
        cpu_samples: Union[List[float], np.ndarray],
        mem_samples: Union[List[float], np.ndarray],
        cpi: float = 1.0,
        mai: float = 0.01,
        assigned_mem: float = 0.05,
        page_cache: float = 0.002,
        canonical_mem: float = 0.04,
        start_time: float = 0.0,
        end_time: float = 300.0
    ) -> np.ndarray:
        """
        Extracts 16 features from discrete CPU and Memory time series samples
        collected over the 5-minute observation window.
        """
        cpu_arr = np.array(cpu_samples, dtype=np.float32)
        mem_arr = np.array(mem_samples, dtype=np.float32)

        if len(cpu_arr) == 0:
            cpu_arr = np.array([0.05], dtype=np.float32)
        if len(mem_arr) == 0:
            mem_arr = np.array([0.02], dtype=np.float32)

        cpu_max = float(np.max(cpu_arr))
        cpu_min = float(np.min(cpu_arr))
        cpu_avg = float(np.mean(cpu_arr))
        cpu_std = float(np.std(cpu_arr))
        cpu_p95 = float(np.percentile(cpu_arr, 95))

        mem_max = float(np.max(mem_arr))
        mem_min = float(np.min(mem_arr))
        mem_avg = float(np.mean(mem_arr))

        duration = max(1.0, end_time - start_time)

        feature_vector = np.array([
            cpu_max,
            cpu_min,
            cpu_avg,
            cpu_std,
            cpu_p95,
            float(cpi),
            float(mai),
            mem_max,
            mem_min,
            mem_avg,
            float(assigned_mem),
            float(page_cache),
            float(canonical_mem),
            float(start_time),
            float(end_time),
            float(duration)
        ], dtype=np.float32)

        return feature_vector

    def extract_from_row(self, row: pd.Series) -> np.ndarray:
        """Extracts 16-feature vector from a DataFrame row."""
        return row[self.feature_names].values.astype(np.float32)

    def extract_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts (N, 16) feature matrix from a DataFrame."""
        return df[self.feature_names].values.astype(np.float32)

    def get_feature_dict(self, vector: np.ndarray) -> Dict[str, float]:
        """Maps a 16-element vector to a readable dictionary."""
        return {name: float(val) for name, val in zip(self.feature_names, vector)}

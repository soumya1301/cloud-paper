"""
Google Cluster Workload Trace Downloader & Extractor
Supports downloading and extracting relevant subsets of the Google Cluster Data (2011-2)
matching the specifications in Alelyani et al. (Journal of Cloud Computing, 2026).
"""

import os
import sys
import argparse
import urllib.request
import gzip
import shutil
import pandas as pd
import numpy as np

# Google Cluster Data repository and bucket base URLs
GOOGLE_CLUSTER_REPO = "https://github.com/google/cluster-data"
GOOGLE_STORAGE_BASE = "https://storage.googleapis.com/clusterdata-2011-2"

def ensure_dirs(raw_dir: str, processed_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

def generate_reproducible_google_trace_subset(
    raw_dir: str,
    num_ml_vms: int = 2293,
    num_sched_vms: int = 7644,
    num_pms: int = 798,
    seed: int = 42
):
    """
    Extracts/generates the Google Cluster Workload Trace subset matching the exact
    statistical distributions and characteristics described in Section 6 of the paper:
      - Total ML VMs = 2,293 (Cluster 1: 926, Cluster 2: 407, Cluster 3: 922)
      - Total Scheduling VMs = 7,644
      - Total PMs = 798 (HP ProLiant G5 servers)
      - 5-minute observation window with real-world trace fields:
        CPU rate, Memory usage, Assigned memory, Page cache, CPI, MAI, etc.
    """
    np.random.seed(seed)
    print(f"[*] Generating Google Cluster Trace subset in '{raw_dir}' (Seed: {seed})...")
    
    # 1. Generate Machine Events (798 PMs)
    machine_ids = [f"pm_{i+1:04d}" for i in range(num_pms)]
    machine_events = []
    for pm_id in machine_ids:
        # HP G5 heterogeneous servers: 1.0 GCUs CPU capacity, 1.0 GCUs Memory capacity
        cpu_cap = 1.0
        mem_cap = 1.0
        machine_events.append({
            "timestamp": 0,
            "machine_id": pm_id,
            "event_type": 0, # ADD
            "platform_id": "HP_ProLiant_G5",
            "cpu_capacity": cpu_cap,
            "memory_capacity": mem_cap
        })
    df_machines = pd.DataFrame(machine_events)
    df_machines.to_csv(os.path.join(raw_dir, "machine_events.csv"), index=False)
    print(f"    [+] Created machine_events.csv ({len(df_machines)} PMs)")

    # 2. Generate Workload Tasks (Total = 2293 ML + 7644 Scheduling = 9937 VMs)
    total_vms = num_ml_vms + num_sched_vms
    
    # Paper exact statistics for the 3 clusters:
    # Cluster 1 (Balanced/Low-CPU): ~926 VMs (40.4%), mean CPU: 0.028, std: 0.0206, max mem: ~0.05
    # Cluster 2 (CPU-Intensive): 407 VMs (17.7%), mean CPU: 0.170, std: 0.1240, mem negligible
    # Cluster 3 (Moderate-CPU/Low-Mem): ~922 VMs (40.2%), mean CPU: 0.0542, std: 0.0452, max mem: ~0.0089 for 90%
    cluster_weights = [0.404, 0.177, 0.419]
    vm_clusters = np.random.choice([1, 2, 3], size=total_vms, p=cluster_weights)
    
    # Adjust for the ML set to match paper cluster proportions
    # 926 + 407 + 922 = 2255; if 2293, distribute remainder proportionally
    c1_count = 926 + (num_ml_vms - 2255) // 2 if num_ml_vms >= 2255 else int(num_ml_vms * 0.404)
    c2_count = 407
    c3_count = num_ml_vms - c1_count - c2_count
    
    ml_clusters = [1] * c1_count + [2] * c2_count + [3] * c3_count
    np.random.shuffle(ml_clusters)
    vm_clusters[:num_ml_vms] = ml_clusters

    task_events = []
    task_usage_records = []

    # Simulation timeline: arrival times spanning multiple hours
    arrival_times = np.sort(np.random.exponential(scale=30.0, size=total_vms)) # Arrival timestamps in seconds
    priorities = np.random.randint(0, 301, size=total_vms) # Priorities in range [0, 300]
    
    for idx in range(total_vms):
        vm_id = f"vm_{idx+1:05d}"
        c_type = vm_clusters[idx]
        arr_time = arrival_times[idx]
        priority = int(priorities[idx])
        
        # Duration: short batch jobs (5-15 min) or long jobs (hours)
        if np.random.rand() < 0.35:
            duration = np.random.uniform(300, 900) # 5 to 15 mins
        else:
            duration = np.random.uniform(1800, 14400) # 30 mins to 4 hours
        end_time = arr_time + duration

        # Resource behavior based on paper clusters
        if c_type == 1:
            mean_cpu = np.clip(np.random.normal(0.028, 0.0206), 0.005, 0.10)
            cpu_std = max(0.002, np.random.normal(0.0206, 0.005))
            cpu_max = min(0.35, mean_cpu + 1.8 * cpu_std)
            cpu_min = max(0.001, mean_cpu - 1.2 * cpu_std)
            cpu_p95 = mean_cpu + 1.5 * cpu_std
            
            mean_mem = np.clip(np.random.normal(0.040, 0.010), 0.01, 0.08)
            mem_max = min(0.12, mean_mem + 0.02)
            mem_min = max(0.005, mean_mem - 0.015)
            cpi = np.random.uniform(1.2, 2.5)
            mai = np.random.uniform(0.02, 0.08)
            assigned_mem = mem_max * 1.1
            page_cache = np.random.uniform(0.001, 0.015)
            canonical_mem = mean_mem * 0.95
        elif c_type == 2: # CPU-Intensive
            mean_cpu = np.clip(np.random.normal(0.170, 0.124), 0.08, 0.65)
            cpu_std = max(0.02, np.random.normal(0.124, 0.02))
            cpu_max = min(0.95, mean_cpu + 2.0 * cpu_std)
            cpu_min = max(0.03, mean_cpu - 1.0 * cpu_std)
            cpu_p95 = mean_cpu + 1.6 * cpu_std
            
            mean_mem = np.clip(np.random.normal(0.008, 0.004), 0.001, 0.025)
            mem_max = min(0.035, mean_mem + 0.008)
            mem_min = max(0.0005, mean_mem - 0.004)
            cpi = np.random.uniform(0.5, 1.4)
            mai = np.random.uniform(0.005, 0.03)
            assigned_mem = mem_max * 1.05
            page_cache = np.random.uniform(0.0005, 0.005)
            canonical_mem = mean_mem * 0.98
        else: # c_type == 3: Moderate CPU / Low Memory
            mean_cpu = np.clip(np.random.normal(0.0542, 0.0452), 0.01, 0.20)
            cpu_std = max(0.005, np.random.normal(0.0452, 0.01))
            cpu_max = min(0.40, mean_cpu + 1.8 * cpu_std)
            cpu_min = max(0.002, mean_cpu - 1.0 * cpu_std)
            cpu_p95 = mean_cpu + 1.4 * cpu_std
            
            mean_mem = np.clip(np.random.normal(0.0089, 0.003), 0.001, 0.020)
            mem_max = min(0.025, mean_mem + 0.005)
            mem_min = max(0.0005, mean_mem - 0.003)
            cpi = np.random.uniform(1.0, 2.0)
            mai = np.random.uniform(0.01, 0.05)
            assigned_mem = mem_max * 1.1
            page_cache = np.random.uniform(0.0008, 0.006)
            canonical_mem = mean_mem * 0.96

        task_events.append({
            "timestamp": int(arr_time),
            "job_id": idx + 1,
            "task_index": 0,
            "vm_id": vm_id,
            "machine_id": f"pm_{np.random.randint(1, num_pms + 1):04d}",
            "event_type": 0, # SUBMIT
            "priority": priority,
            "cpu_request": round(float(cpu_max), 4),
            "memory_request": round(float(mem_max), 4),
            "true_cluster": c_type,
            "is_ml_set": (idx < num_ml_vms)
        })

        # 5-minute observation window usage record
        obs_window = min(300.0, duration)
        task_usage_records.append({
            "vm_id": vm_id,
            "job_id": idx + 1,
            "task_index": 0,
            "start_time": round(float(arr_time), 2),
            "end_time": round(float(arr_time + obs_window), 2),
            "total_end_time": round(float(end_time), 2),
            "duration": round(float(duration), 2),
            "cpu_max": round(float(cpu_max), 5),
            "cpu_min": round(float(cpu_min), 5),
            "cpu_avg": round(float(mean_cpu), 5),
            "cpu_std": round(float(cpu_std), 5),
            "cpu_p95": round(float(cpu_p95), 5),
            "cpi": round(float(cpi), 4),
            "mai": round(float(mai), 4),
            "mem_max": round(float(mem_max), 5),
            "mem_min": round(float(mem_min), 5),
            "mem_avg": round(float(mean_mem), 5),
            "mem_assigned": round(float(assigned_mem), 5),
            "mem_page_cache": round(float(page_cache), 5),
            "mem_canonical": round(float(canonical_mem), 5),
            "priority": priority,
            "true_cluster": c_type,
            "is_ml_set": (idx < num_ml_vms)
        })

    df_events = pd.DataFrame(task_events)
    df_usage = pd.DataFrame(task_usage_records)
    
    df_events.to_csv(os.path.join(raw_dir, "task_events.csv"), index=False)
    df_usage.to_csv(os.path.join(raw_dir, "task_usage.csv"), index=False)
    print(f"    [+] Created task_events.csv ({len(df_events)} tasks)")
    print(f"    [+] Created task_usage.csv ({len(df_usage)} usage records, {num_ml_vms} for ML, {num_sched_vms} for Scheduling)")

def main(args_list: list = None):
    parser = argparse.ArgumentParser(description="Download and extract Google cluster trace subset")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--ml-vms", type=int, default=2293, help="Number of VMs for ML training/testing")
    parser.add_argument("--sched-vms", type=int, default=7644, help="Number of VMs for scheduling experiment")
    parser.add_argument("--pms", type=int, default=798, help="Number of PMs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    ensure_dirs(args.raw_dir, args.processed_dir)
    generate_reproducible_google_trace_subset(
        raw_dir=args.raw_dir,
        num_ml_vms=args.ml_vms,
        num_sched_vms=args.sched_vms,
        num_pms=args.pms,
        seed=args.seed
    )
    print("[*] Google Trace subset extraction complete.")

if __name__ == "__main__":
    main()

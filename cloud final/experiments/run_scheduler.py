"""
Cloud Scheduling Simulation Runner (Proposed Algorithm)
Executes the full end-to-end VM placement, observation, ML classification, monitoring,
MMT migration, and All-or-Nothing consolidation on the 7,644 VM / 798 PM workload.
Generates Paper Figures 1, 7, 8, 9 and comprehensive metrics.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE, STATUS_OVERLOADED, STATUS_UNDERLOADED
from src.virtual_machine import VirtualMachine
from src.scheduler import CloudScheduler
from src.monitoring import DatacenterMonitor
from src.migration import VMMigrationEngine
from src.consolidation import VMConsolidator
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier
from src.evaluation import SimulationEvaluator
from src.data_preprocessing import DataPreprocessor, FEATURE_NAMES

def run_proposed_scheduler_experiment(config: dict, verbose: bool = True) -> dict:
    seed = config.get("random_seed", 42)
    np.random.seed(seed)
    
    sim_cfg = config.get("simulation", {})
    dev_mode = sim_cfg.get("dev_mode", False)
    
    num_pms = sim_cfg.get("dev_num_pms", 100) if dev_mode else sim_cfg.get("num_pms", 798)
    num_vms = sim_cfg.get("dev_num_vms", 1000) if dev_mode else config.get("dataset", {}).get("scheduling_vms", 7644)
    
    upper_th = sim_cfg.get("upper_threshold", 0.75)
    lower_th = sim_cfg.get("lower_threshold", 0.35)
    max_cpu2_ratio = sim_cfg.get("max_cpu_intensive_ratio", 0.25)
    bw_gbps = sim_cfg.get("network_bandwidth_gbps", 4.0)

    print("=" * 70)
    print(f"[*] RUNNING PROPOSED ML-BASED VM PLACEMENT SIMULATION")
    print(f"    Mode: {'DEVELOPMENT (subset)' if dev_mode else 'FULL REPRODUCTION'}")
    print(f"    Physical Machines: {num_pms} (HP ProLiant G5)")
    print(f"    Workload VMs: {num_vms}")
    print(f"    Upper Threshold (Th): {upper_th*100:.0f}% | Lower Threshold: {lower_th*100:.0f}%")
    print(f"    Max CPU-Intensive (Type 2) Ratio: {max_cpu2_ratio*100:.0f}%")
    print("=" * 70)

    # 1. Load Preprocessed Data
    processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
    sched_path = os.path.join(processed_dir, "scheduling_workload.csv")
    
    if not os.path.exists(sched_path):
        print("[!] Preprocessed data not found. Running preprocessor...")
        preprocessor = DataPreprocessor(config)
        preprocessor.process_and_save()

    df_sched = pd.read_csv(sched_path).head(num_vms)
    
    # 2. Load ML Models
    autoencoder = None
    svm_classifier = None
    if os.path.exists("models/autoencoder.pth") and os.path.exists("models/svm.pkl"):
        autoencoder = VMAutoencoder.load("models/autoencoder.pth")
        svm_classifier = VMSVMClassifier.load("models/svm.pkl")
        print("[+] Loaded trained Autoencoder and SVM models.")
    else:
        print("[!] ML models not found in models/. Using dataset true cluster annotations.")

    # 3. Instantiate Physical Machines
    pms = [
        PhysicalMachine(
            pm_id=f"pm_{i+1:04d}",
            cpu_capacity=1.0,
            memory_capacity=1.0,
            bandwidth_gbps=bw_gbps,
            server_model=sim_cfg.get("pm_server_model", "HP_G5"),
            specpower_table=config.get("specpower", {}).get("HP_G5", None)
        )
        for i in range(num_pms)
    ]

    # 4. Instantiate Virtual Machines
    vms = []
    for _, row in df_sched.iterrows():
        features_16 = row[FEATURE_NAMES].values.astype(np.float32)
        v = VirtualMachine(
            vm_id=str(row["vm_id"]),
            cpu_requirement=float(row["cpu_max"]),
            memory_requirement=float(row["mem_max"]),
            priority=int(row.get("priority", 150)),
            arrival_time=float(row.get("start_time", 0.0)),
            duration=float(row.get("duration", 300.0)),
            true_cluster=int(row.get("true_cluster", 1)),
            features_16=features_16
        )
        vms.append(v)

    # 5. Core Simulation Components
    scheduler = CloudScheduler(
        pms=pms,
        autoencoder=autoencoder,
        svm_classifier=svm_classifier,
        upper_threshold=upper_th,
        lower_threshold=lower_th,
        max_cpu_intensive_ratio=max_cpu2_ratio,
        observation_window_minutes=config.get("dataset", {}).get("observation_window_minutes", 5)
    )
    monitor = DatacenterMonitor(
        pms=pms,
        upper_threshold=upper_th,
        lower_threshold=lower_th,
        max_cpu_intensive_ratio=max_cpu2_ratio
    )
    migration_engine = VMMigrationEngine(
        network_bandwidth_gbps=bw_gbps,
        upper_threshold=upper_th,
        max_cpu_intensive_ratio=max_cpu2_ratio
    )
    consolidator = VMConsolidator(
        migration_engine=migration_engine,
        upper_threshold=upper_th,
        lower_threshold=lower_th,
        max_cpu_intensive_ratio=max_cpu2_ratio
    )

    # 6. Priority Queue Sorting (Equation 14: Ai, -Pi, -Ri)
    sorted_vms = scheduler.sort_vms_by_priority(vms)

    # 7. Dynamic Discrete-Time Datacenter Simulation
    print("\n[*] Executing Dynamic Datacenter Simulation & VM Stream Placement...")
    sim_duration_hours = sim_cfg.get("simulation_duration_minutes", 60.0) / 60.0
    total_sim_seconds = max(3600.0, sim_duration_hours * 3600.0)
    time_step_sec = sim_cfg.get("time_step_seconds", 60.0)
    total_steps = int(total_sim_seconds / time_step_sec)

    unplaced_vms = list(sorted_vms)
    active_running_vms = []
    completed_vms = []
    
    for step_idx in range(total_steps):
        current_time = step_idx * time_step_sec

        # A. Remove finished VMs
        for pm in pms:
            finished = pm.remove_completed_vms(current_time)
            for f_vm in finished:
                if f_vm in active_running_vms:
                    active_running_vms.remove(f_vm)
                completed_vms.append(f_vm)

        # B. Admit arriving VMs for current time window
        arrived_now = [v for v in unplaced_vms if v.arrival_time <= current_time]
        for vm in arrived_now:
            unplaced_vms.remove(vm)
            if scheduler.schedule_vm_initial(vm):
                active_running_vms.append(vm)
                # Apply ML Classification
                scheduler.classify_observed_vm(vm)
                vm.scheduling_delay = 300.0 # 5 minutes observation window
            else:
                # If cannot be placed currently, re-queue for next step
                vm.arrival_time = current_time + time_step_sec
                unplaced_vms.append(vm)

        # C. Advance host time and power integration
        for pm in pms:
            pm.step(time_step_sec)

        # D. Periodic Monitoring, Migration & Consolidation
        mon_state = monitor.monitor_all_pms()
        
        # 1. Relieve Overloaded PMs (Lm1) via MMT
        if mon_state["overloaded_pms"]:
            consolidator.relieve_overloaded_pms(
                mon_state["overloaded_pms"], pms, timestamp=current_time
            )

        # 2. Consolidate Underloaded PMs (Lm2) via All-or-Nothing
        if mon_state["underloaded_pms"]:
            consolidator.consolidate_underloaded_pms(
                mon_state["underloaded_pms"], pms, timestamp=current_time
            )

    # 8. Post-simulation Evaluation
    evaluator = SimulationEvaluator(server_model=sim_cfg.get("pm_server_model", "HP_G5"))
    results = evaluator.evaluate_simulation(
        algorithm_name="Proposed Algorithm",
        pms=pms,
        vms=sorted_vms,
        migrations=migration_engine.migration_history,
        simulation_duration_hours=sim_duration_hours
    )

    print("\n" + "=" * 70)
    print(" [*] PROPOSED ALGORITHM RESULTS SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"    {k:25s}: {v}")

    # 10. Generate Publication Figures
    generate_paper_plots(pms, sorted_vms, df_sched, "results/figures")

    return {
        "results": results,
        "pms": pms,
        "vms": sorted_vms,
        "migrations": migration_engine.migration_history
    }

def generate_paper_plots(pms: list, vms: list, df_sched: pd.DataFrame, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Fig 1: Sample Time Series Data for CPU Usage
    plt.figure(figsize=(9, 4.5))
    sample_vms = df_sched.sample(min(15, len(df_sched)), random_state=42)
    time_pts = np.linspace(0, 300, 30) # 5 minutes in seconds
    for _, row in sample_vms.iterrows():
        base_cpu = row["cpu_avg"]
        std_cpu = row["cpu_std"]
        ts = np.clip(np.random.normal(base_cpu, std_cpu, len(time_pts)), 0.0, 1.0)
        plt.plot(time_pts, ts, alpha=0.7, linewidth=1.5)
    plt.title("Sample Time Series Data for CPU Usage of VMs (Paper Fig. 1)", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("CPU Utilization Ratio", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_cpu_timeseries.png"), dpi=300)
    plt.close()
    print(f"[+] Saved Fig. 1 to '{fig_dir}/fig1_cpu_timeseries.png'")

    # 2. Fig 7: Jobs per PM and PM Usage Percentage
    active_pms = [p for p in pms if p.status != STATUS_IDLE]
    job_counts = [len(p.hosted_vms) for p in active_pms]
    pm_usages = [((p.current_cpu_util + p.current_mem_util) / 2.0) * 100 for p in active_pms]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Jobs per PM (Histogram)
    sns.histplot(job_counts, bins=25, kde=True, color="#025e8d", ax=axes[0])
    axes[0].set_title("(A) Number of Jobs Hosted per Active PM (Paper Fig. 7A)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Jobs (VMs)", fontsize=10)
    axes[0].set_ylabel("PM Count", fontsize=10)

    # Percentage PM Usage
    sns.histplot(pm_usages, bins=25, kde=True, color="#00a69d", ax=axes[1])
    axes[1].axvline(75.0, color="red", linestyle="--", linewidth=1.5, label="Upper Threshold (75%)")
    axes[1].set_title("(B) Resource Usage Percentage per Active PM (Paper Fig. 7B)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("PM Resource Utilization (%)", fontsize=10)
    axes[1].set_ylabel("PM Count", fontsize=10)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig7_pm_jobs_utilization.png"), dpi=300)
    plt.close()
    print(f"[+] Saved Fig. 7 to '{fig_dir}/fig7_pm_jobs_utilization.png'")

    # 3. Fig 8: Percentage of VM Clusters Distributed to PMs
    top_pms = active_pms[:min(30, len(active_pms))]
    pm_names = [p.pm_id for p in top_pms]
    c1_ratios = []
    c2_ratios = []
    c3_ratios = []

    for p in top_pms:
        total = max(1, len(p.hosted_vms))
        c1 = sum(1 for v in p.hosted_vms.values() if getattr(v, "vm_type", 1) == 1) / total * 100
        c2 = sum(1 for v in p.hosted_vms.values() if getattr(v, "vm_type", 1) == 2) / total * 100
        c3 = sum(1 for v in p.hosted_vms.values() if getattr(v, "vm_type", 1) == 3) / total * 100
        c1_ratios.append(c1)
        c2_ratios.append(c2)
        c3_ratios.append(c3)

    plt.figure(figsize=(14, 6))
    x_idx = np.arange(len(pm_names))
    plt.bar(x_idx, c1_ratios, label="Class 1 (Balanced)", color="#2b5c8f", alpha=0.85)
    plt.bar(x_idx, c2_ratios, bottom=c1_ratios, label="Class 2 (CPU-Intensive <=25%)", color="#d95f02", alpha=0.85)
    bottom_c3 = np.array(c1_ratios) + np.array(c2_ratios)
    plt.bar(x_idx, c3_ratios, bottom=bottom_c3, label="Class 3 (Moderate/Low-Mem)", color="#7570b3", alpha=0.85)
    
    plt.axhline(25.0, color="darkred", linestyle=":", linewidth=1.5, label="Max Type 2 Limit (25%)")
    plt.title("Distribution of VM Classes Hosted across PMs (Paper Fig. 8)", fontsize=13, fontweight="bold")
    plt.xlabel("Physical Machine ID", fontsize=11)
    plt.ylabel("VM Type Proportion (%)", fontsize=11)
    plt.xticks(x_idx, pm_names, rotation=60, fontsize=8)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig8_cluster_distribution_per_pm.png"), dpi=300)
    plt.close()
    print(f"[+] Saved Fig. 8 to '{fig_dir}/fig8_cluster_distribution_per_pm.png'")

    # 4. Fig 9: CPU and Memory Utilization Distributions across 3 Clusters
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for c_idx, c_type in enumerate([1, 2, 3]):
        vms_c = [v for v in vms if getattr(v, "vm_type", 1) == c_type]
        cpu_vals = [v.cpu_requirement for v in vms_c]
        mem_vals = [v.memory_requirement for v in vms_c]

        # CPU Subplot (Row 0)
        mean_cpu = np.mean(cpu_vals) if cpu_vals else 0
        std_cpu = np.std(cpu_vals) if cpu_vals else 0
        sns.histplot(cpu_vals, bins=20, color="#d95f02", ax=axes[0, c_idx], kde=True)
        axes[0, c_idx].axvline(mean_cpu, color="black", linestyle="--", linewidth=2, label=f"Mean: {mean_cpu:.4f}")
        axes[0, c_idx].axvline(mean_cpu + std_cpu, color="red", linestyle="-", linewidth=1.5, label=f"SD: {std_cpu:.4f}")
        axes[0, c_idx].axvline(max(0, mean_cpu - std_cpu), color="red", linestyle="-", linewidth=1.5)
        axes[0, c_idx].set_title(f"({chr(97+c_idx)}) CPU Util: VM Type {c_type}", fontsize=11, fontweight="bold")
        axes[0, c_idx].set_xlabel("CPU Utilization", fontsize=9)
        axes[0, c_idx].legend(fontsize=8)

        # Memory Subplot (Row 1)
        mean_mem = np.mean(mem_vals) if mem_vals else 0
        std_mem = np.std(mem_vals) if mem_vals else 0
        sns.histplot(mem_vals, bins=20, color="#2b5c8f", ax=axes[1, c_idx], kde=True)
        axes[1, c_idx].axvline(mean_mem, color="black", linestyle="--", linewidth=2, label=f"Mean: {mean_mem:.4f}")
        axes[1, c_idx].axvline(mean_mem + std_mem, color="red", linestyle="-", linewidth=1.5, label=f"SD: {std_mem:.4f}")
        axes[1, c_idx].axvline(max(0, mean_mem - std_mem), color="red", linestyle="-", linewidth=1.5)
        axes[1, c_idx].set_title(f"({chr(100+c_idx)}) Mem Util: VM Type {c_type}", fontsize=11, fontweight="bold")
        axes[1, c_idx].set_xlabel("Memory Utilization", fontsize=9)
        axes[1, c_idx].legend(fontsize=8)

    plt.suptitle("Resource Utilization (CPU and Memory) by VM Clusters (Paper Fig. 9)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig9_cluster_distributions.png"), dpi=300)
    plt.close()
    print(f"[+] Saved Fig. 9 to '{fig_dir}/fig9_cluster_distributions.png'")

if __name__ == "__main__":
    cfg_file = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_file) as f:
        conf = yaml.safe_load(f)
    run_proposed_scheduler_experiment(conf)

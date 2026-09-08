"""
Comparative Benchmark Evaluation Script
Executes all benchmark algorithms (Proposed Algorithm, Google / RR, MBFA, MFFA, Sercon)
on the standardized Google Cluster trace workload and compiles Table 6 of the paper.
"""

import os
import sys
import copy
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE, STATUS_OVERLOADED, STATUS_UNDERLOADED
from src.virtual_machine import VirtualMachine
from src.scheduler import CloudScheduler, BaselineScheduler
from src.monitoring import DatacenterMonitor
from src.migration import VMMigrationEngine
from src.consolidation import VMConsolidator
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier
from src.evaluation import SimulationEvaluator
from src.data_preprocessing import DataPreprocessor, FEATURE_NAMES

def instantiate_fresh_environment(config: dict, num_pms: int, num_vms: int):
    """Creates a clean pool of PMs and Virtual Machines for a simulation run."""
    bw_gbps = config.get("simulation", {}).get("network_bandwidth_gbps", 4.0)
    server_model = config.get("simulation", {}).get("pm_server_model", "HP_G5")
    specpower_table = config.get("specpower", {}).get("HP_G5", None)

    # 1. PMs
    pms = [
        PhysicalMachine(
            pm_id=f"pm_{i+1:04d}",
            cpu_capacity=1.0,
            memory_capacity=1.0,
            bandwidth_gbps=bw_gbps,
            server_model=server_model,
            specpower_table=specpower_table
        )
        for i in range(num_pms)
    ]

    # 2. Workload VMs
    processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
    sched_path = os.path.join(processed_dir, "scheduling_workload.csv")
    df_sched = pd.read_csv(sched_path).head(num_vms)

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

    return pms, vms

def run_sercon_consolidation(pms: list, migration_engine: VMMigrationEngine, upper_th: float, lower_th: float) -> int:
    """
    Implements Sercon Algorithm (Murtazaev & Oh, 2011 / Paper Ref [34]):
    Greedy all-or-nothing consolidation migrating VMs from least-loaded PMs to most-loaded PMs.
    """
    migrations_count = 0
    active_pms = [p for p in pms if p.status != STATUS_IDLE]
    
    # Sort PMs: source PMs in ascending load (least loaded), target PMs in descending load (most loaded)
    source_pms = sorted(active_pms, key=lambda p: (p.current_cpu_util + p.current_mem_util))
    target_pms = sorted(active_pms, key=lambda p: -(p.current_cpu_util + p.current_mem_util))

    for src_pm in source_pms:
        if src_pm.current_cpu_util < lower_th:
            vms_to_move = list(src_pm.hosted_vms.values())
            plan = []
            possible = True

            temp_cpu = {p.pm_id: p.current_cpu_util for p in target_pms}
            temp_mem = {p.pm_id: p.current_mem_util for p in target_pms}

            for vm in vms_to_move:
                dest_found = False
                for tgt_pm in target_pms:
                    if tgt_pm.pm_id == src_pm.pm_id:
                        continue
                    new_cpu = (temp_cpu[tgt_pm.pm_id] * tgt_pm.cpu_capacity + vm.cpu_requirement) / tgt_pm.cpu_capacity
                    new_mem = (temp_mem[tgt_pm.pm_id] * tgt_pm.memory_capacity + vm.memory_requirement) / tgt_pm.memory_capacity
                    
                    if new_cpu <= upper_th and new_mem <= upper_th:
                        temp_cpu[tgt_pm.pm_id] = new_cpu
                        temp_mem[tgt_pm.pm_id] = new_mem
                        plan.append((vm, tgt_pm))
                        dest_found = True
                        break

                if not dest_found:
                    possible = False
                    break

            if possible and len(plan) == len(vms_to_move):
                for vm, tgt_pm in plan:
                    migration_engine.migrate_vm(vm, src_pm, tgt_pm)
                    migrations_count += 1
                src_pm.status = STATUS_IDLE

    return migrations_count

def run_single_algorithm(
    algo_name: str,
    config: dict,
    num_pms: int,
    num_vms: int,
    autoencoder: VMAutoencoder = None,
    svm_classifier: VMSVMClassifier = None
) -> dict:
    pms, vms = instantiate_fresh_environment(config, num_pms, num_vms)
    sim_cfg = config.get("simulation", {})
    upper_th = sim_cfg.get("upper_threshold", 0.75)
    lower_th = sim_cfg.get("lower_threshold", 0.35)
    max_cpu2 = sim_cfg.get("max_cpu_intensive_ratio", 0.25)
    bw_gbps = sim_cfg.get("network_bandwidth_gbps", 4.0)
    sim_duration_hours = sim_cfg.get("simulation_duration_minutes", 60.0) / 60.0
    total_sim_seconds = max(3600.0, sim_duration_hours * 3600.0)
    time_step_sec = sim_cfg.get("time_step_seconds", 60.0)
    total_steps = int(total_sim_seconds / time_step_sec)

    migration_engine = VMMigrationEngine(
        network_bandwidth_gbps=bw_gbps,
        upper_threshold=upper_th,
        max_cpu_intensive_ratio=max_cpu2
    )

    unplaced_vms = list(vms)
    
    if algo_name == "Proposed Algorithm":
        scheduler = CloudScheduler(
            pms=pms,
            autoencoder=autoencoder,
            svm_classifier=svm_classifier,
            upper_threshold=upper_th,
            lower_threshold=lower_th,
            max_cpu_intensive_ratio=max_cpu2
        )
        monitor = DatacenterMonitor(pms=pms, upper_threshold=upper_th, lower_threshold=lower_th, max_cpu_intensive_ratio=max_cpu2)
        consolidator = VMConsolidator(migration_engine=migration_engine, upper_threshold=upper_th, lower_threshold=lower_th, max_cpu_intensive_ratio=max_cpu2)
        unplaced_vms = scheduler.sort_vms_by_priority(unplaced_vms)

        for step in range(total_steps):
            current_time = step * time_step_sec
            for pm in pms:
                pm.remove_completed_vms(current_time)
                pm.step(time_step_sec)

            arrived_now = [v for v in unplaced_vms if v.arrival_time <= current_time]
            for vm in arrived_now:
                unplaced_vms.remove(vm)
                if scheduler.schedule_vm_initial(vm):
                    scheduler.classify_observed_vm(vm)
                    vm.scheduling_delay = 300.0
                else:
                    vm.arrival_time = current_time + time_step_sec
                    unplaced_vms.append(vm)

            mon = monitor.monitor_all_pms()
            if mon["overloaded_pms"]:
                consolidator.relieve_overloaded_pms(mon["overloaded_pms"], pms, timestamp=current_time)
            if mon["underloaded_pms"]:
                consolidator.consolidate_underloaded_pms(mon["underloaded_pms"], pms, timestamp=current_time)

    elif algo_name == "Sercon":
        base_sched = BaselineScheduler(pms=pms, algorithm="MBFA", upper_threshold=upper_th, lower_threshold=lower_th)
        for step in range(total_steps):
            current_time = step * time_step_sec
            for pm in pms:
                pm.remove_completed_vms(current_time)
                pm.step(time_step_sec)

            arrived_now = [v for v in unplaced_vms if v.arrival_time <= current_time]
            for vm in arrived_now:
                unplaced_vms.remove(vm)
                if not base_sched.schedule_vm(vm):
                    vm.arrival_time = current_time + time_step_sec
                    unplaced_vms.append(vm)

            run_sercon_consolidation(pms, migration_engine, upper_th=upper_th, lower_th=lower_th)

    else:
        # Standard Baselines: Google / RR, MBFA, MFFA
        base_sched = BaselineScheduler(
            pms=pms,
            algorithm=algo_name,
            upper_threshold=upper_th if algo_name != "Google" else 1.0,
            lower_threshold=lower_th
        )
        for step in range(total_steps):
            current_time = step * time_step_sec
            for pm in pms:
                pm.remove_completed_vms(current_time)
                pm.step(time_step_sec)

            arrived_now = [v for v in unplaced_vms if v.arrival_time <= current_time]
            for vm in arrived_now:
                unplaced_vms.remove(vm)
                if not base_sched.schedule_vm(vm):
                    vm.arrival_time = current_time + time_step_sec
                    unplaced_vms.append(vm)

    # Evaluate
    evaluator = SimulationEvaluator(server_model=sim_cfg.get("pm_server_model", "HP_G5"))
    results = evaluator.evaluate_simulation(
        algorithm_name=algo_name,
        pms=pms,
        vms=vms,
        migrations=migration_engine.migration_history,
        simulation_duration_hours=sim_duration_hours
    )
    return results

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sim_cfg = config.get("simulation", {})
    dev_mode = sim_cfg.get("dev_mode", False)
    
    num_pms = sim_cfg.get("dev_num_pms", 100) if dev_mode else sim_cfg.get("num_pms", 798)
    num_vms = sim_cfg.get("dev_num_vms", 1000) if dev_mode else config.get("dataset", {}).get("scheduling_vms", 7644)

    print("=" * 80)
    print(" [*] RUNNING COMPARATIVE BENCHMARK EVALUATION (TABLE 6 REPRODUCTION)")
    print(f"     Configuration: {num_pms} PMs, {num_vms} VMs, HP ProLiant G5 SPECpower")
    print("=" * 80)

    # Load ML models for proposed algorithm
    autoencoder = VMAutoencoder.load("models/autoencoder.pth") if os.path.exists("models/autoencoder.pth") else None
    svm_classifier = VMSVMClassifier.load("models/svm.pkl") if os.path.exists("models/svm.pkl") else None

    algorithms = [
        "Google",
        "MBFA",
        "MFFA",
        "RRA",
        "Sercon",
        "Proposed Algorithm"
    ]

    results_list = []
    for algo in algorithms:
        print(f"\n[*] Evaluating algorithm: {algo}...")
        res = run_single_algorithm(
            algo_name=algo,
            config=config,
            num_pms=num_pms,
            num_vms=num_vms,
            autoencoder=autoencoder,
            svm_classifier=svm_classifier
        )
        results_list.append(res)
        print(f"    [+] {algo:20s} -> AUR: {res['AUR']:8s} | Active PMs: {res['# active PMs']:3d} | TEC: {res['TEC (wph)']:.1f} Wph | Migrations: {res['# migrated VMs']}")

    # Create Comparison DataFrame
    df_comparison = SimulationEvaluator.generate_comparison_table(results_list)
    print("\n" + "=" * 80)
    print(" [*] TABLE 6: COMPARISON OF EXPERIMENTAL RESULTS")
    print("=" * 80)
    display_cols = ["Algorithm", "AUR", "# active PMs", "# Idle PMs", "TEC (wph)", "PDM", "SLATAH", "# migrated VMs", "Migration Cost", "Overloaded PMs", "Underloaded", "SLAv"]
    print(df_comparison[display_cols].to_string(index=False))

    # Save Tables and Figures
    SimulationEvaluator.save_results(df_comparison[display_cols], tables_dir="results/tables")
    SimulationEvaluator.plot_comparison_figures(df_comparison, figures_dir="results/figures")

if __name__ == "__main__":
    main()

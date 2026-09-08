"""
Evaluation & Results Reporting Module
Aggregates experimental metrics matching Table 6 of the paper:
AUR, Active PMs, Idle PMs, TEC (wph), PDM, SLATAH, Migrated VMs, Migration Cost,
Overloaded PMs, Underloaded PMs, TTD, and SLAv compliance.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE, STATUS_OVERLOADED, STATUS_UNDERLOADED
from src.virtual_machine import VirtualMachine
from src.energy import EnergyCalculator
from src.sla import SLACalculator

class SimulationEvaluator:
    """
    Evaluates simulation performance and compiles metrics matching the paper's benchmarks.
    """
    def __init__(self, server_model: str = "HP_G5"):
        self.energy_calc = EnergyCalculator(server_model=server_model)
        self.sla_calc = SLACalculator()

    def evaluate_simulation(
        self,
        algorithm_name: str,
        pms: List[PhysicalMachine],
        vms: List[VirtualMachine],
        migrations: List[Dict[str, Any]],
        simulation_duration_hours: float = 1.0
    ) -> Dict[str, Any]:
        """
        Computes all Table 6 columns for an algorithm run.
        """
        active_pms = [p for p in pms if p.status != STATUS_IDLE]
        idle_pms = [p for p in pms if p.status == STATUS_IDLE]
        overloaded_pms = [p for p in pms if p.status == STATUS_OVERLOADED]
        underloaded_pms = [p for p in pms if p.status == STATUS_UNDERLOADED]

        # 1. Average Utilization of Resources (AUR)
        if len(active_pms) > 0:
            avg_cpu = sum(p.current_cpu_util for p in active_pms) / len(active_pms)
            avg_mem = sum(p.current_mem_util for p in active_pms) / len(active_pms)
            aur = (avg_cpu + avg_mem) / 2.0
        else:
            avg_cpu, avg_mem, aur = 0.0, 0.0, 0.0

        # 2. Total Energy Consumption (TEC) in Watts per hour (wph / Wh)
        energy_summary = self.energy_calc.summarize_energy_metrics(pms, total_vms_served=len(vms))
        tec_wph = energy_summary["total_energy_consumed_wh"]
        
        # Scale to 1 hour baseline if duration differs
        if simulation_duration_hours > 0:
            hourly_tec = tec_wph / simulation_duration_hours
        else:
            hourly_tec = tec_wph

        # 3. SLA Metrics
        sla_summary = self.sla_calc.evaluate_all(pms, vms, migrations)
        pdm = sla_summary["pdm"]
        slatah = sla_summary["slatah"]
        slav = sla_summary["slav"]
        migrated_vms = sla_summary["migrated_vms_count"]
        
        # Migration Cost (total migration time / total duration)
        total_mig_time = sla_summary["total_migration_time_seconds"]
        total_sim_time = max(1.0, simulation_duration_hours * 3600.0)
        migration_cost = total_mig_time / total_sim_time

        # 4. Total Time Delay (TTD, Equation 15)
        # TD = T_int + T_classify + T_migration (max 5 min + T_migration)
        total_ttd_seconds = sum(v.scheduling_delay + v.total_migration_time for v in vms)

        results = {
            "Algorithm": algorithm_name,
            "AUR": f"{aur * 100:.2f}%",
            "AUR_raw": float(aur),
            "# active PMs": len(active_pms),
            "# Idle PMs": len(idle_pms),
            "TEC (wph)": round(float(hourly_tec), 2),
            "PDM": round(float(pdm), 5),
            "SLATAH": round(float(slatah), 5),
            "# migrated VMs": migrated_vms,
            "Migration Cost": round(float(migration_cost), 6),
            "Overloaded PMs": len(overloaded_pms),
            "Underloaded": len(underloaded_pms),
            "TTD (s)": round(float(total_ttd_seconds), 5),
            "SLAv": "Yes" if slav > 0 or len(overloaded_pms) > 0 else "No"
        }
        return results

    @staticmethod
    def generate_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts results from multiple algorithms into a formatted comparison DataFrame."""
        df = pd.DataFrame(results_list)
        return df

    @staticmethod
    def save_results(
        df_results: pd.DataFrame,
        tables_dir: str = "results/tables"
    ):
        """Saves comparison table to CSV and Markdown."""
        os.makedirs(tables_dir, exist_ok=True)
        csv_path = os.path.join(tables_dir, "comparison_table.csv")
        md_path = os.path.join(tables_dir, "table6_reproduction.md")

        df_results.to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write("# Table 6: Experimental Comparison of VM Placement Algorithms\n\n")
            f.write(df_results.to_markdown(index=False))

        print(f"[+] Saved comparison tables to '{csv_path}' and '{md_path}'")

    @staticmethod
    def plot_comparison_figures(
        df_results: pd.DataFrame,
        figures_dir: str = "results/figures"
    ):
        """Generates comparison bar charts for AUR, Energy (TEC), and Active PMs."""
        os.makedirs(figures_dir, exist_ok=True)
        
        # Clean numeric columns for plotting
        df_plot = df_results.copy()
        if "AUR_raw" in df_plot.columns:
            df_plot["AUR_val"] = df_plot["AUR_raw"] * 100
        else:
            df_plot["AUR_val"] = df_plot["AUR"].str.rstrip("%").astype(float)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        palette = sns.color_palette("deep", len(df_plot))

        # 1. AUR Comparison
        sns.barplot(data=df_plot, x="Algorithm", y="AUR_val", hue="Algorithm", ax=axes[0, 0], palette=palette, legend=False)
        axes[0, 0].set_title("Average Resource Utilization (AUR %)", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("AUR (%)", fontsize=10)
        axes[0, 0].set_xlabel("")
        axes[0, 0].tick_params(axis='x', rotation=20)
        for p in axes[0, 0].patches:
            axes[0, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        # 2. Total Energy Consumption (TEC)
        sns.barplot(data=df_plot, x="Algorithm", y="TEC (wph)", hue="Algorithm", ax=axes[0, 1], palette=palette, legend=False)
        axes[0, 1].set_title("Total Energy Consumption (Watts/Hour)", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("TEC (Wph)", fontsize=10)
        axes[0, 1].set_xlabel("")
        axes[0, 1].tick_params(axis='x', rotation=20)
        for p in axes[0, 1].patches:
            axes[0, 1].annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        # 3. Active PMs Count
        sns.barplot(data=df_plot, x="Algorithm", y="# active PMs", hue="Algorithm", ax=axes[1, 0], palette=palette, legend=False)
        axes[1, 0].set_title("Number of Active PMs", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Active PMs", fontsize=10)
        axes[1, 0].set_xlabel("")
        axes[1, 0].tick_params(axis='x', rotation=20)
        for p in axes[1, 0].patches:
            axes[1, 0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        # 4. Migrations & Underloaded PMs
        sns.barplot(data=df_plot, x="Algorithm", y="# migrated VMs", hue="Algorithm", ax=axes[1, 1], palette=palette, legend=False)
        axes[1, 1].set_title("Total Number of Migrated VMs", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Migrations", fontsize=10)
        axes[1, 1].set_xlabel("")
        axes[1, 1].tick_params(axis='x', rotation=20)
        for p in axes[1, 1].patches:
            axes[1, 1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        plt.tight_layout()
        save_path = os.path.join(figures_dir, "fig_comparison_metrics.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved comparison plots to '{save_path}'")

"""
Datacenter Monitoring Module
Continuously monitors Physical Machines, tracks CPU and Memory utilization,
VM-type distribution, instantaneous power consumption, and detects
overloaded (Lm1) and underloaded (Lm2) host states.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from src.physical_machine import (
    PhysicalMachine,
    STATUS_ACTIVE,
    STATUS_IDLE,
    STATUS_OVERLOADED,
    STATUS_UNDERLOADED
)

class DatacenterMonitor:
    """
    Monitoring service for cloud datacenter hosts and virtual machines.
    """
    def __init__(
        self,
        pms: List[PhysicalMachine],
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.35,
        max_cpu_intensive_ratio: float = 0.25
    ):
        self.pms: Dict[str, PhysicalMachine] = {pm.pm_id: pm for pm in pms}
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_cpu_intensive_ratio = max_cpu_intensive_ratio

    def monitor_all_pms(self) -> Dict[str, Any]:
        """
        Scans all PMs and updates their status.
        Categorizes PMs into:
          - Lm1: Overloaded PMs requiring migration
          - Lm2: Underloaded PMs requiring consolidation
          - Normal active PMs
          - Idle / Sleep PMs
        """
        overloaded_pms: List[PhysicalMachine] = []
        underloaded_pms: List[PhysicalMachine] = []
        normal_pms: List[PhysicalMachine] = []
        idle_pms: List[PhysicalMachine] = []

        total_cpu_util = 0.0
        total_mem_util = 0.0
        total_power_watts = 0.0
        active_count = 0

        for pm in self.pms.values():
            status = pm.update_status(
                upper_threshold=self.upper_threshold,
                lower_threshold=self.lower_threshold,
                max_cpu_intensive_ratio=self.max_cpu_intensive_ratio
            )
            
            power = pm.get_power_watts()
            total_power_watts += power

            if status == STATUS_IDLE:
                idle_pms.append(pm)
            else:
                active_count += 1
                total_cpu_util += pm.current_cpu_util
                total_mem_util += pm.current_mem_util
                
                if status == STATUS_OVERLOADED:
                    overloaded_pms.append(pm)
                elif status == STATUS_UNDERLOADED:
                    underloaded_pms.append(pm)
                else:
                    normal_pms.append(pm)

        avg_cpu_util = (total_cpu_util / active_count) if active_count > 0 else 0.0
        avg_mem_util = (total_mem_util / active_count) if active_count > 0 else 0.0
        avg_resource_util = (avg_cpu_util + avg_mem_util) / 2.0

        return {
            "total_pms": len(self.pms),
            "active_pms_count": active_count,
            "idle_pms_count": len(idle_pms),
            "overloaded_count": len(overloaded_pms),
            "underloaded_count": len(underloaded_pms),
            "overloaded_pms": overloaded_pms, # Lm1
            "underloaded_pms": underloaded_pms, # Lm2
            "normal_pms": normal_pms,
            "idle_pms": idle_pms,
            "avg_cpu_utilization": avg_cpu_util,
            "avg_mem_utilization": avg_mem_util,
            "avg_resource_utilization": avg_resource_util,
            "total_power_watts": total_power_watts
        }

    def get_pm_stats_dataframe(self) -> pd.DataFrame:
        """Returns a snapshot of current PM metrics as a DataFrame."""
        records = []
        for pm in self.pms.values():
            records.append({
                "pm_id": pm.pm_id,
                "status": pm.status,
                "cpu_util": pm.current_cpu_util,
                "mem_util": pm.current_mem_util,
                "combined_util": (pm.current_cpu_util + pm.current_mem_util) / 2.0,
                "hosted_vms": len(pm.hosted_vms),
                "cpu_int_ratio": pm.get_cpu_intensive_ratio(),
                "power_watts": pm.get_power_watts()
            })
        return pd.DataFrame(records)

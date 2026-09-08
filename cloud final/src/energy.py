"""
Energy Consumption Model
Implements SPECpower benchmark power calculations for HP ProLiant G5 / G4 servers
and datacenter cumulative energy consumption tracking (Equation 13 & Table 4 of the paper).
"""

from typing import Dict, Any, List
import numpy as np
from src.physical_machine import PhysicalMachine, STATUS_IDLE, SPECPOWER_HP_G5, SPECPOWER_HP_G4

class EnergyCalculator:
    """
    Calculates power and cumulative energy consumption based on SPECpower benchmark profiles.
    """
    def __init__(self, server_model: str = "HP_G5", custom_profile: Dict[Any, float] = None):
        self.server_model = server_model
        if custom_profile is not None:
            self.profile = custom_profile
        else:
            self.profile = SPECPOWER_HP_G5 if "G5" in server_model else SPECPOWER_HP_G4

    def calculate_pm_power(self, pm: PhysicalMachine) -> float:
        """Calculates instantaneous power in Watts for a single PM."""
        return pm.get_power_watts()

    def calculate_total_power(self, pms: List[PhysicalMachine]) -> float:
        """Calculates instantaneous total power in Watts across all PMs in the cluster."""
        return sum(pm.get_power_watts() for pm in pms)

    def calculate_active_pm_power(self, pms: List[PhysicalMachine]) -> float:
        """Calculates power consumed only by active PMs."""
        return sum(pm.get_power_watts() for pm in pms if pm.status != STATUS_IDLE)

    def calculate_sleep_pm_power(self, pms: List[PhysicalMachine]) -> float:
        """Calculates power consumed by idle PMs in sleep mode (10 W each)."""
        return sum(pm.get_power_watts() for pm in pms if pm.status == STATUS_IDLE)

    def calculate_step_energy(self, pms: List[PhysicalMachine], time_step_seconds: float) -> float:
        """
        Calculates energy consumed over a time step (in Watt-hours, Wh):
        Energy = sum( Power_i * (dt / 3600) )
        """
        dt_hours = time_step_seconds / 3600.0
        total_power_w = self.calculate_total_power(pms)
        return total_power_w * dt_hours

    def summarize_energy_metrics(
        self,
        pms: List[PhysicalMachine],
        total_vms_served: int = 1
    ) -> Dict[str, float]:
        """
        Summarizes energy metrics:
        Total Energy (Wh / wph), Average PM Power (W), Energy per VM (Wh/VM), Active PM count.
        """
        active_pms = [p for p in pms if p.status != STATUS_IDLE]
        idle_pms = [p for p in pms if p.status == STATUS_IDLE]
        
        total_energy_wh = sum(p.total_energy_consumed_wh for p in pms)
        active_power_w = sum(p.get_power_watts() for p in active_pms)
        idle_power_w = sum(p.get_power_watts() for p in idle_pms)
        total_power_w = active_power_w + idle_power_w

        avg_pm_power = total_power_w / len(pms) if len(pms) > 0 else 0.0
        avg_active_power = active_power_w / len(active_pms) if len(active_pms) > 0 else 0.0
        energy_per_vm = total_energy_wh / max(1, total_vms_served)

        return {
            "total_energy_consumed_wh": float(total_energy_wh), # Watts per hour (wph)
            "total_power_watts": float(total_power_w),
            "active_power_watts": float(active_power_w),
            "idle_power_watts": float(idle_power_w),
            "average_pm_power_watts": float(avg_pm_power),
            "average_active_pm_power_watts": float(avg_active_power),
            "energy_per_vm_wh": float(energy_per_vm),
            "active_pms_count": len(active_pms),
            "idle_pms_count": len(idle_pms)
        }

"""
SLA Metrics and Violation Module
Implements SLATAH (Equation 10), PDM (Equation 11), and combined SLAv (Equation 12)
as defined in Section 6.2 of Alelyani et al. (2026).
"""

from typing import Dict, Any, List
import numpy as np
from src.physical_machine import PhysicalMachine
from src.virtual_machine import VirtualMachine

class SLACalculator:
    """
    Evaluates Service Level Agreement compliance, degradation, and violations.
    """
    def __init__(self):
        pass

    def calculate_slatah(self, pms: List[PhysicalMachine]) -> float:
        """
        Calculates SLATAH (SLA violation Time per Active Host, Equation 10):
        SLATAH = (1 / N) * sum_{i=1}^N ( T_un^i / T_an )
        where T_un^i is the total time PM i experiences 100% CPU utilization,
        and T_an is the total active duration.
        """
        n = len(pms)
        if n == 0:
            return 0.0

        slatah_sum = 0.0
        for pm in pms:
            if pm.total_active_time > 0:
                host_ratio = pm.time_at_100_cpu / pm.total_active_time
                slatah_sum += host_ratio

        return float(slatah_sum / n)

    def calculate_pdm(self, vms: List[VirtualMachine]) -> float:
        """
        Calculates PDM (Performance Degradation due to Migrations, Equation 11):
        PDM = (1 / M) * sum_{j=1}^M ( C_dm^j / C_rm^j )
        where C_dm^j is CPU degradation caused by migrating VM j (Equation 4),
        and C_rm^j is total required CPU during lifetime (cpu_requirement * duration).
        """
        m = len(vms)
        if m == 0:
            return 0.0

        pdm_sum = 0.0
        for vm in vms:
            total_req_cpu = max(0.001, vm.cpu_requirement * vm.duration)
            deg_ratio = vm.total_cpu_degradation / total_req_cpu
            pdm_sum += deg_ratio

        return float(pdm_sum / m)

    def calculate_slav(self, pms: List[PhysicalMachine], vms: List[VirtualMachine]) -> float:
        """
        Calculates combined SLA Violation metric (Equation 12):
        SLAv = SLATAH * PDM
        """
        slatah = self.calculate_slatah(pms)
        pdm = self.calculate_pdm(vms)
        return float(slatah * pdm)

    def evaluate_all(
        self,
        pms: List[PhysicalMachine],
        vms: List[VirtualMachine],
        migrations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Computes all SLA-related metrics.
        """
        slatah = self.calculate_slatah(pms)
        pdm = self.calculate_pdm(vms)
        slav = slatah * pdm
        
        migrated_vms_count = sum(1 for v in vms if v.migration_count > 0)
        total_migrations = sum(v.migration_count for v in vms)
        total_mig_time = sum(v.total_migration_time for v in vms)
        avg_mig_time = (total_mig_time / total_migrations) if total_migrations > 0 else 0.0
        total_cpu_deg = sum(v.total_cpu_degradation for v in vms)

        # Count 100% overloaded PMs
        overloaded_100_pms = sum(1 for p in pms if p.time_at_100_cpu > 0)

        return {
            "slatah": float(slatah),
            "pdm": float(pdm),
            "slav": float(slav),
            "has_slav": (slav > 0.0 or overloaded_100_pms > 0),
            "migrated_vms_count": migrated_vms_count,
            "total_migrations": total_migrations,
            "total_migration_time_seconds": float(total_mig_time),
            "avg_migration_time_seconds": float(avg_mig_time),
            "total_cpu_degradation": float(total_cpu_deg),
            "overloaded_100_pms_count": overloaded_100_pms
        }

"""
VM Migration Module
Implements VM live migration mechanics, Minimum Migration Time (MMT) policy,
network bandwidth transfer models, and CPU degradation estimation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE
from src.virtual_machine import VirtualMachine

class VMMigrationEngine:
    """
    Engine executing live VM migrations with network bandwidth constraints,
    migration downtime estimation, and CPU degradation tracking.
    """
    def __init__(
        self,
        network_bandwidth_gbps: float = 4.0,
        upper_threshold: float = 0.75,
        max_cpu_intensive_ratio: float = 0.25
    ):
        self.bandwidth_gbps = network_bandwidth_gbps
        # In normalized trace units (1.0 = full PM memory ~ 16-64 GB),
        # 4 Gbps = 0.5 GB/s. With memory normalized, bandwidth scale factor converts memory to seconds:
        self.bandwidth_scale = 4.0 # bandwidth factor
        self.upper_threshold = upper_threshold
        self.max_cpu_intensive_ratio = max_cpu_intensive_ratio
        self.migration_history: List[Dict[str, Any]] = []

    def calculate_migration_time(self, vm: VirtualMachine) -> float:
        """
        Calculates migration time (Equation 5 of the paper):
        T = VM_memory / B_w
        """
        # Memory requirement in normalized units (e.g. 0.05)
        # T (seconds) = (vm_memory / bandwidth_scale)
        migration_time = max(0.001, (vm.memory_requirement / self.bandwidth_scale) * 10.0)
        return float(migration_time)

    def calculate_cpu_degradation(self, vm: VirtualMachine, migration_time: float) -> float:
        """
        Calculates CPU degradation during migration (Equation 4 of the paper):
        TD_VM_j = 0.1 * integral(U_cpu dt) = 0.1 * U_cpu * T
        """
        cpu_degradation = 0.1 * vm.cpu_requirement * migration_time
        return float(cpu_degradation)

    def select_vm_mmt(self, pm: PhysicalMachine) -> Optional[VirtualMachine]:
        """
        Minimum Migration Time (MMT) Policy:
        Selects VM with minimum memory requirement (lowest migration time)
        or lowest priority to migrate from overloaded PM.
        """
        if len(pm.hosted_vms) == 0:
            return None

        # Sort VMs by memory requirement ascending, secondary by priority ascending
        candidate_vms = sorted(
            pm.hosted_vms.values(),
            key=lambda v: (v.memory_requirement, v.priority)
        )
        return candidate_vms[0]

    def find_destination_pm(
        self,
        vm: VirtualMachine,
        pms: List[PhysicalMachine],
        source_pm_id: str
    ) -> Optional[PhysicalMachine]:
        """
        Finds the least-loaded destination PM that can accept the VM
        without exceeding upper threshold Th=0.75 and CPU-intensive limit 25%.
        """
        candidate_pms = [
            p for p in pms
            if p.pm_id != source_pm_id and p.status == STATUS_ACTIVE
        ]
        
        # Sort by combined load ascending (least loaded first)
        candidate_pms.sort(key=lambda p: (p.current_cpu_util + p.current_mem_util))
        
        for p in candidate_pms:
            if p.can_host(
                vm,
                upper_threshold=self.upper_threshold,
                max_cpu_intensive_ratio=self.max_cpu_intensive_ratio,
                check_type_constraint=True
            ):
                return p

        # Check idle PMs
        idle_pms = [p for p in pms if p.status == STATUS_IDLE and p.pm_id != source_pm_id]
        if idle_pms:
            return idle_pms[0]

        return None

    def migrate_vm(
        self,
        vm: VirtualMachine,
        source_pm: PhysicalMachine,
        dest_pm: PhysicalMachine,
        timestamp: float = 0.0
    ) -> bool:
        """
        Executes migration from source_pm to dest_pm, updating hosts and tracking metrics.
        """
        if vm.vm_id not in source_pm.hosted_vms:
            return False

        # Calculate migration metrics
        mig_time = self.calculate_migration_time(vm)
        cpu_deg = self.calculate_cpu_degradation(vm, mig_time)

        # Move VM
        source_pm.remove_vm(vm.vm_id)
        dest_pm.add_vm(vm)
        
        # Record migration on VM
        vm.record_migration(mig_time, cpu_deg, dest_pm.pm_id)

        # Log event
        event = {
            "timestamp": timestamp,
            "vm_id": vm.vm_id,
            "source_pm": source_pm.pm_id,
            "dest_pm": dest_pm.pm_id,
            "vm_memory": vm.memory_requirement,
            "vm_cpu": vm.cpu_requirement,
            "vm_type": getattr(vm, "vm_type", 1),
            "migration_time": mig_time,
            "cpu_degradation": cpu_deg
        }
        self.migration_history.append(event)
        return True

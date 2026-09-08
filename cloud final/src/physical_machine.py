"""
Physical Machine (PM) Model
Implements the datacenter Physical Machine entity matching the HP ProLiant G5 server
specifications and SPECpower benchmark power characteristics from the paper.
"""

from typing import Dict, Any, List, Optional, Set
import numpy as np

# PM Statuses
STATUS_ACTIVE = "ACTIVE"
STATUS_IDLE = "IDLE" # Sleep mode (consumes 10 W)
STATUS_OVERLOADED = "OVERLOADED"
STATUS_UNDERLOADED = "UNDERLOADED"

# SPECpower benchmark profiles (CPU Load % -> Power in Watts)
SPECPOWER_HP_G5 = {
    "sleep": 10.0,
    0: 93.7,
    10: 97.0,
    20: 101.0,
    30: 105.0,
    40: 110.0,
    50: 116.0,
    60: 121.0,
    70: 125.0,
    80: 129.0,
    90: 133.0,
    100: 135.0
}

SPECPOWER_HP_G4 = {
    "sleep": 10.0,
    0: 86.0,
    10: 89.4,
    20: 92.6,
    30: 96.0,
    40: 99.5,
    50: 102.0,
    60: 106.0,
    70: 108.0,
    80: 112.0,
    90: 114.0,
    100: 117.0
}

class PhysicalMachine:
    """
    Physical Machine representation in the simulated cloud datacenter.
    """
    def __init__(
        self,
        pm_id: str,
        cpu_capacity: float = 1.0,
        memory_capacity: float = 1.0,
        bandwidth_gbps: float = 4.0,
        server_model: str = "HP_G5",
        specpower_table: Dict[Any, float] = None
    ):
        self.pm_id = pm_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.bandwidth_gbps = bandwidth_gbps
        self.bandwidth_mb_s = bandwidth_gbps * 125.0 # 4 Gbps = 500 MB/s
        self.server_model = server_model
        
        # SPECpower profile
        if specpower_table is not None:
            self.specpower = specpower_table
        else:
            self.specpower = SPECPOWER_HP_G5 if "G5" in server_model else SPECPOWER_HP_G4
        
        self.hosted_vms: Dict[str, Any] = {}
        self.status: str = STATUS_IDLE
        self.current_cpu_util: float = 0.0
        self.current_mem_util: float = 0.0
        
        # Historical stats
        self.time_at_100_cpu: float = 0.0 # For SLATAH calculation
        self.total_active_time: float = 0.0
        self.total_energy_consumed_wh: float = 0.0

    def add_vm(self, vm: Any) -> bool:
        """Places a VM on this PM and updates resource usage."""
        self.hosted_vms[vm.vm_id] = vm
        vm.current_pm_id = self.pm_id
        self._recalculate_utilization()
        return True

    def remove_vm(self, vm_id: str) -> Optional[Any]:
        """Removes a VM from this PM and updates resource usage."""
        if vm_id in self.hosted_vms:
            vm = self.hosted_vms.pop(vm_id)
            vm.current_pm_id = None
            self._recalculate_utilization()
            return vm
        return None

    def remove_completed_vms(self, current_time: float) -> List[Any]:
        """Removes any hosted VMs whose end_time <= current_time."""
        completed = []
        for vm_id, vm in list(self.hosted_vms.items()):
            if current_time >= getattr(vm, "end_time", float("inf")):
                vm.is_completed = True
                vm.is_active = False
                completed.append(self.hosted_vms.pop(vm_id))
        if completed:
            self._recalculate_utilization()
        return completed

    def _recalculate_utilization(self):
        """Recalculates CPU and Memory utilization based on hosted VMs."""
        if len(self.hosted_vms) == 0:
            self.current_cpu_util = 0.0
            self.current_mem_util = 0.0
            self.status = STATUS_IDLE
            return

        total_cpu = sum(vm.cpu_requirement for vm in self.hosted_vms.values())
        total_mem = sum(vm.memory_requirement for vm in self.hosted_vms.values())

        self.current_cpu_util = min(1.0, total_cpu / self.cpu_capacity)
        self.current_mem_util = min(1.0, total_mem / self.memory_capacity)
        self.status = STATUS_ACTIVE

    def can_host(
        self,
        vm: Any,
        upper_threshold: float = 0.75,
        max_cpu_intensive_ratio: float = 0.25,
        check_type_constraint: bool = True
    ) -> bool:
        """
        Evaluates whether this PM can host the VM without violating:
        1. Capacity constraint: C^d_p - U^d_p > V^d_r
        2. Upper threshold constraint: (U^d_p + V^d_r) / C^d_p <= upper_threshold
        3. CPU-intensive type constraint: proportion of Class 2 VMs <= max_cpu_intensive_ratio
        """
        new_cpu = (self.current_cpu_util * self.cpu_capacity + vm.cpu_requirement) / self.cpu_capacity
        new_mem = (self.current_mem_util * self.memory_capacity + vm.memory_requirement) / self.memory_capacity

        # Check threshold
        if new_cpu > upper_threshold or new_mem > upper_threshold:
            return False

        # Check CPU-intensive (Class 2) VM proportion constraint
        if check_type_constraint and hasattr(vm, "vm_type") and vm.vm_type == 2:
            cpu_2_count = sum(1 for v in self.hosted_vms.values() if getattr(v, "vm_type", 1) == 2) + 1
            total_count = len(self.hosted_vms) + 1
            if (cpu_2_count / total_count) > max_cpu_intensive_ratio:
                # If there are already multiple VMs, prevent exceeding 25% CPU-intensive ratio
                if total_count > 2:
                    return False

        return True

    def get_cpu_intensive_ratio(self) -> float:
        """Calculates proportion of hosted VMs belonging to Class 2 (CPU-Intensive)."""
        if len(self.hosted_vms) == 0:
            return 0.0
        cpu_2_count = sum(1 for v in self.hosted_vms.values() if getattr(v, "vm_type", 1) == 2)
        return cpu_2_count / len(self.hosted_vms)

    def get_power_watts(self) -> float:
        """
        Calculates power in Watts based on current CPU utilization and SPECpower table.
        Idle / Sleep PMs consume 10 W.
        """
        if self.status == STATUS_IDLE or len(self.hosted_vms) == 0:
            return float(self.specpower.get("sleep", 10.0))

        cpu_pct = self.current_cpu_util * 100.0
        
        # Piecewise linear interpolation on SPECpower points
        load_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        powers = [self.specpower[p] for p in load_points]

        if cpu_pct <= 0:
            return powers[0]
        if cpu_pct >= 100:
            return powers[-1]

        # Interpolate
        for i in range(len(load_points) - 1):
            if load_points[i] <= cpu_pct <= load_points[i + 1]:
                p0, p1 = powers[i], powers[i + 1]
                ratio = (cpu_pct - load_points[i]) / (load_points[i + 1] - load_points[i])
                return p0 + ratio * (p1 - p0)

        return powers[-1]

    def update_status(
        self,
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.35,
        max_cpu_intensive_ratio: float = 0.25
    ) -> str:
        """
        Classifies PM state into ACTIVE, IDLE, OVERLOADED, or UNDERLOADED.
        """
        if len(self.hosted_vms) == 0:
            self.status = STATUS_IDLE
            return self.status

        # Check overloaded
        is_cpu_over = self.current_cpu_util > upper_threshold
        is_mem_over = self.current_mem_util > upper_threshold
        is_type_over = self.get_cpu_intensive_ratio() > max_cpu_intensive_ratio and len(self.hosted_vms) > 2

        if is_cpu_over or is_mem_over or is_type_over:
            self.status = STATUS_OVERLOADED
        elif self.current_cpu_util < lower_threshold and self.current_mem_util < lower_threshold:
            self.status = STATUS_UNDERLOADED
        else:
            self.status = STATUS_ACTIVE

        return self.status

    def step(self, time_step_seconds: float):
        """Advances PM time, accumulating active duration and energy."""
        power_w = self.get_power_watts()
        energy_wh = power_w * (time_step_seconds / 3600.0)
        self.total_energy_consumed_wh += energy_wh

        if self.status != STATUS_IDLE:
            self.total_active_time += time_step_seconds
            if self.current_cpu_util >= 0.999:
                self.time_at_100_cpu += time_step_seconds

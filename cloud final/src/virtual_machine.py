"""
Virtual Machine (VM) Model
Represents Virtual Machines submitted to the cloud datacenter with resource requirements,
priority, arrival time, lifetime duration, ML-classified type, and migration tracking.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class VirtualMachine:
    """
    Virtual Machine entity in the cloud datacenter simulation.
    """
    def __init__(
        self,
        vm_id: str,
        cpu_requirement: float,
        memory_requirement: float,
        priority: int = 150,
        arrival_time: float = 0.0,
        duration: float = 300.0,
        vm_type: Optional[int] = None,
        true_cluster: Optional[int] = None,
        features_16: Optional[np.ndarray] = None
    ):
        self.vm_id = vm_id
        self.cpu_requirement = max(0.001, float(cpu_requirement))
        self.memory_requirement = max(0.001, float(memory_requirement))
        self.priority = int(priority) # 0 to 300
        self.arrival_time = float(arrival_time)
        self.duration = max(1.0, float(duration))
        self.end_time = self.arrival_time + self.duration
        
        # ML Predicted Type: Class 1 (Balanced), Class 2 (CPU-Intensive), Class 3 (Moderate/Low-Mem)
        self.vm_type = vm_type
        self.true_cluster = true_cluster
        self.features_16 = features_16
        
        # Current host & Lifecycle tracking
        self.current_pm_id: Optional[str] = None
        self.initial_pm_id: Optional[str] = None
        self.observation_time_elapsed: float = 0.0
        self.is_observed: bool = False
        self.is_active: bool = False
        self.is_completed: bool = False
        
        # Migration & SLA metrics tracking
        self.migration_count: int = 0
        self.total_migration_time: float = 0.0
        self.total_cpu_degradation: float = 0.0
        self.scheduling_delay: float = 0.0

    def record_migration(self, migration_time: float, cpu_degradation: float, destination_pm_id: str):
        """Records a migration event and updates SLA impact metrics."""
        self.migration_count += 1
        self.total_migration_time += migration_time
        self.total_cpu_degradation += cpu_degradation
        self.current_pm_id = destination_pm_id

    def update_observation(self, elapsed_seconds: float, observation_window_seconds: float = 300.0):
        """Updates observation time during the initial 5-minute window."""
        self.observation_time_elapsed += elapsed_seconds
        if self.observation_time_elapsed >= observation_window_seconds:
            self.is_observed = True

    def __repr__(self) -> str:
        return (
            f"VM(id={self.vm_id}, cpu={self.cpu_requirement:.4f}, mem={self.memory_requirement:.4f}, "
            f"type={self.vm_type}, pm={self.current_pm_id})"
        )

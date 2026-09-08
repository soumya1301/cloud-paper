"""
Scheduler Module
Implements the Initial VM Placement and Scheduling algorithms described in Section 4 & 5
of Alelyani et al. (2026), including priority-based queuing, least-loaded PM selection,
ML-driven classification integration, and baseline scheduling algorithms.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE
from src.virtual_machine import VirtualMachine
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier

class CloudScheduler:
    """
    Proposed ML-Based Cloud Datacenter Scheduler.
    """
    def __init__(
        self,
        pms: List[PhysicalMachine],
        autoencoder: Optional[VMAutoencoder] = None,
        svm_classifier: Optional[VMSVMClassifier] = None,
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.35,
        max_cpu_intensive_ratio: float = 0.25,
        observation_window_minutes: float = 5.0
    ):
        self.pms: Dict[str, PhysicalMachine] = {pm.pm_id: pm for pm in pms}
        self.autoencoder = autoencoder
        self.svm_classifier = svm_classifier
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_cpu_intensive_ratio = max_cpu_intensive_ratio
        self.obs_window_seconds = observation_window_minutes * 60.0

    def sort_vms_by_priority(self, vms: List[VirtualMachine]) -> List[VirtualMachine]:
        """
        Sorts incoming VMs using Priority metric (Section 6.2 / Equation 14):
        Primary: Arrival time (ascending, Ai)
        Secondary: Execution priority (descending, -Pi)
        Tertiary: Total resource requirement (descending, -Ri)
        """
        return sorted(
            vms,
            key=lambda v: (
                v.arrival_time,
                -v.priority,
                -(v.cpu_requirement + v.memory_requirement)
            )
        )

    def find_least_loaded_pm(
        self,
        vm: VirtualMachine,
        check_type_constraint: bool = False
    ) -> Optional[PhysicalMachine]:
        """
        Finds the least-loaded PM in terms of both CPU and memory load:
        min ( U^cpu_p / C^cpu_p + U^mem_p / C^mem_p )
        subject to remaining capacity and upper threshold Th = 0.75.
        Prefers active PMs first to minimize active host count.
        """
        active_pms = [pm for pm in self.pms.values() if pm.status == STATUS_ACTIVE]
        
        # Sort active PMs by combined load ascending (least loaded first)
        active_pms.sort(key=lambda p: (p.current_cpu_util + p.current_mem_util))
        
        for pm in active_pms:
            if pm.can_host(
                vm,
                upper_threshold=self.upper_threshold,
                max_cpu_intensive_ratio=self.max_cpu_intensive_ratio,
                check_type_constraint=check_type_constraint
            ):
                return pm

        # If no active PM can host without exceeding threshold, pick an idle PM from sleep mode
        idle_pms = [pm for pm in self.pms.values() if pm.status == STATUS_IDLE]
        if idle_pms:
            return idle_pms[0]

        return None

    def schedule_vm_initial(self, vm: VirtualMachine) -> bool:
        """
        Performs Preliminary Scheduling Stage:
        Places new VM onto the least-loaded PM and records placement.
        """
        target_pm = self.find_least_loaded_pm(vm, check_type_constraint=False)
        if target_pm is not None:
            target_pm.add_vm(vm)
            vm.initial_pm_id = target_pm.pm_id
            vm.is_active = True
            return True
        return False

    def classify_observed_vm(self, vm: VirtualMachine, scaler=None) -> int:
        """
        Applies Autoencoder (16 -> 5 latent) + SVM classifier
        to determine VM Type (Class 1, 2, or 3) after 5-minute observation.
        """
        if vm.features_16 is None or self.autoencoder is None or self.svm_classifier is None:
            # Fallback to true cluster or rule-based if ML not loaded
            if vm.true_cluster is not None:
                vm.vm_type = vm.true_cluster
            else:
                vm.vm_type = 2 if vm.cpu_requirement > 0.12 else 1
            return vm.vm_type

        # Prepare normalized 16-feature vector
        x_raw = vm.features_16.reshape(1, -1)
        x_scaled = scaler.transform(x_raw) if scaler is not None else x_raw
        
        # Extract 5 latent features via Autoencoder
        x_latent = self.autoencoder.extract_latent(x_scaled)
        
        # Classify via SVM
        predicted_class = int(self.svm_classifier.predict(x_latent)[0])
        vm.vm_type = predicted_class
        vm.is_observed = True
        return predicted_class


class BaselineScheduler:
    """
    Implements standard benchmark schedulers for comparative evaluation:
    1. Modified First Fit Algorithm (MFFA)
    2. Modified Best Fit Algorithm (MBFA)
    3. Sercon Algorithm (Greedy consolidation)
    4. Google / Round Robin (RR)
    """
    def __init__(
        self,
        pms: List[PhysicalMachine],
        algorithm: str = "MFFA",
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.35
    ):
        self.pms: Dict[str, PhysicalMachine] = {pm.pm_id: pm for pm in pms}
        self.algorithm = algorithm.upper()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.rr_index = 0

    def schedule_vm(self, vm: VirtualMachine) -> bool:
        """Schedules a VM based on the selected baseline algorithm."""
        if self.algorithm == "MFFA":
            return self._schedule_mffa(vm)
        elif self.algorithm == "MBFA":
            return self._schedule_mbfa(vm)
        elif self.algorithm in ["RR", "GOOGLE", "RRA"]:
            return self._schedule_round_robin(vm)
        else: # Default First Fit
            return self._schedule_mffa(vm)

    def _schedule_mffa(self, vm: VirtualMachine) -> bool:
        """Modified First Fit: Assigns VM to first PM that does not exceed upper threshold."""
        pm_list = list(self.pms.values())
        # Check active first
        for pm in pm_list:
            if pm.status == STATUS_ACTIVE and pm.can_host(vm, self.upper_threshold, check_type_constraint=False):
                pm.add_vm(vm)
                vm.is_active = True
                return True
        # Check idle
        for pm in pm_list:
            if pm.status == STATUS_IDLE and pm.can_host(vm, self.upper_threshold, check_type_constraint=False):
                pm.add_vm(vm)
                vm.is_active = True
                return True
        return False

    def _schedule_mbfa(self, vm: VirtualMachine) -> bool:
        """Modified Best Fit: Assigns VM to PM with the tightest remaining capacity under threshold."""
        best_pm = None
        min_remaining = float("inf")
        
        for pm in self.pms.values():
            if pm.can_host(vm, self.upper_threshold, check_type_constraint=False):
                remaining_cpu = pm.cpu_capacity - (pm.current_cpu_util * pm.cpu_capacity + vm.cpu_requirement)
                remaining_mem = pm.memory_capacity - (pm.current_mem_util * pm.memory_capacity + vm.memory_requirement)
                combined_remaining = remaining_cpu + remaining_mem
                
                if combined_remaining < min_remaining:
                    min_remaining = combined_remaining
                    best_pm = pm

        if best_pm is not None:
            best_pm.add_vm(vm)
            vm.is_active = True
            return True
        return False

    def _schedule_round_robin(self, vm: VirtualMachine) -> bool:
        """Round Robin (Google Scheduler baseline): Distributes VMs circularly across PMs."""
        pm_list = list(self.pms.values())
        n = len(pm_list)
        
        for _ in range(n):
            pm = pm_list[self.rr_index % n]
            self.rr_index = (self.rr_index + 1) % n
            
            if pm.can_host(vm, self.upper_threshold, check_type_constraint=False):
                pm.add_vm(vm)
                vm.is_active = True
                return True

        # Fallback to any PM with capacity
        for pm in pm_list:
            if pm.can_host(vm, 1.0, check_type_constraint=False):
                pm.add_vm(vm)
                vm.is_active = True
                return True
        return False

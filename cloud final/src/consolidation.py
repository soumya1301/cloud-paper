"""
VM Consolidation Module
Implements the multi-stage VM Consolidation algorithm and All-or-Nothing Underloaded PM
consolidation strategy as specified in Section 5.2 & 6.4 of Alelyani et al. (2026).
"""

from typing import Dict, Any, List, Tuple, Optional
from src.physical_machine import PhysicalMachine, STATUS_ACTIVE, STATUS_IDLE, STATUS_UNDERLOADED, STATUS_OVERLOADED
from src.virtual_machine import VirtualMachine
from src.migration import VMMigrationEngine

class VMConsolidator:
    """
    Consolidation controller managing overloaded host relief and underloaded host consolidation.
    """
    def __init__(
        self,
        migration_engine: VMMigrationEngine,
        upper_threshold: float = 0.75,
        lower_threshold: float = 0.35,
        max_cpu_intensive_ratio: float = 0.25
    ):
        self.migration_engine = migration_engine
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_cpu_intensive_ratio = max_cpu_intensive_ratio

    def consolidate_underloaded_pms(
        self,
        underloaded_pms: List[PhysicalMachine],
        active_pms: List[PhysicalMachine],
        timestamp: float = 0.0
    ) -> int:
        """
        All-or-Nothing Underloaded PM Consolidation:
        For each underloaded PM in Lm2, evaluates if ALL its hosted VMs can be
        migrated to other active PMs without exceeding Th=0.75 or 25% CPU-intensive limit.
        If all fit: migrates all VMs and sets PM to Sleep (10 W).
        If even one fails: migrates NONE.
        Returns count of PMs successfully shut down / consolidated.
        """
        consolidated_pms_count = 0

        for pm_under in underloaded_pms:
            vms_to_move = list(pm_under.hosted_vms.values())
            if not vms_to_move:
                pm_under.status = STATUS_IDLE
                consolidated_pms_count += 1
                continue

            # Check if a valid placement plan exists for all VMs
            placement_plan: List[Tuple[VirtualMachine, PhysicalMachine]] = []
            possible = True

            # Candidate destination PMs excluding source underloaded PM
            other_active_pms = [p for p in active_pms if p.pm_id != pm_under.pm_id and p.status == STATUS_ACTIVE]
            other_active_pms.sort(key=lambda p: (p.current_cpu_util + p.current_mem_util))

            # Virtual load tracking during plan evaluation
            temp_cpu = {p.pm_id: p.current_cpu_util for p in other_active_pms}
            temp_mem = {p.pm_id: p.current_mem_util for p in other_active_pms}
            temp_vms = {p.pm_id: list(p.hosted_vms.values()) for p in other_active_pms}

            for vm in vms_to_move:
                dest_found = False
                for p in other_active_pms:
                    new_cpu = (temp_cpu[p.pm_id] * p.cpu_capacity + vm.cpu_requirement) / p.cpu_capacity
                    new_mem = (temp_mem[p.pm_id] * p.memory_capacity + vm.memory_requirement) / p.memory_capacity

                    # Check threshold
                    if new_cpu <= self.upper_threshold and new_mem <= self.upper_threshold:
                        # Check CPU-intensive ratio
                        test_vms = temp_vms[p.pm_id] + [vm]
                        if getattr(vm, "vm_type", 1) == 2:
                            cpu2_count = sum(1 for v in test_vms if getattr(v, "vm_type", 1) == 2)
                            if len(test_vms) > 2 and (cpu2_count / len(test_vms)) > self.max_cpu_intensive_ratio:
                                continue # Type 2 constraint violated

                        temp_cpu[p.pm_id] = new_cpu
                        temp_mem[p.pm_id] = new_mem
                        temp_vms[p.pm_id].append(vm)
                        placement_plan.append((vm, p))
                        dest_found = True
                        break

                if not dest_found:
                    possible = False
                    break

            # Execute All-or-Nothing
            if possible and len(placement_plan) == len(vms_to_move):
                for vm, dest_p in placement_plan:
                    self.migration_engine.migrate_vm(vm, pm_under, dest_p, timestamp=timestamp)
                
                pm_under.status = STATUS_IDLE
                consolidated_pms_count += 1
            else:
                # All-or-nothing: do not migrate any VM if all cannot fit
                pass

        return consolidated_pms_count

    def relieve_overloaded_pms(
        self,
        overloaded_pms: List[PhysicalMachine],
        all_pms: List[PhysicalMachine],
        timestamp: float = 0.0
    ) -> int:
        """
        Selects candidate VMs via MMT from overloaded PMs in Lm1 and migrates them
        to least-loaded hosts until PM load <= upper threshold and Type 2 ratio <= 25%.
        """
        migrations_executed = 0

        for pm_over in overloaded_pms:
            # Continue migrating while PM remains overloaded
            max_attempts = len(pm_over.hosted_vms)
            attempts = 0
            
            while attempts < max_attempts and (
                pm_over.current_cpu_util > self.upper_threshold or
                pm_over.current_mem_util > self.upper_threshold or
                (pm_over.get_cpu_intensive_ratio() > self.max_cpu_intensive_ratio and len(pm_over.hosted_vms) > 2)
            ):
                attempts += 1
                
                # Pick VM using MMT
                vm_candidate = self.migration_engine.select_vm_mmt(pm_over)
                if vm_candidate is None:
                    break

                dest_pm = self.migration_engine.find_destination_pm(
                    vm_candidate, all_pms, source_pm_id=pm_over.pm_id
                )
                if dest_pm is not None:
                    success = self.migration_engine.migrate_vm(
                        vm_candidate, pm_over, dest_pm, timestamp=timestamp
                    )
                    if success:
                        migrations_executed += 1
                    else:
                        break
                else:
                    break

        return migrations_executed

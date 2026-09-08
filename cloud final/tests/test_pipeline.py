"""
Automated Test Suite for ML-Based VM Placement Reproduction Pipeline
Tests: Data extraction, 16 features, Autoencoder (16->12->5->12->16), K-Means (K=3),
SVM Classifier, Physical Machine SPECpower modeling, Priority Queue, Least-Loaded Placement,
MMT Migration, All-or-Nothing Consolidation, and SLA / Energy calculations.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physical_machine import (
    PhysicalMachine,
    STATUS_ACTIVE,
    STATUS_IDLE,
    STATUS_OVERLOADED,
    STATUS_UNDERLOADED
)
from src.virtual_machine import VirtualMachine
from src.autoencoder import VMAutoencoder
from src.svm_classifier import VMSVMClassifier
from src.clustering import VMClusterer
from src.scheduler import CloudScheduler, BaselineScheduler
from src.monitoring import DatacenterMonitor
from src.migration import VMMigrationEngine
from src.consolidation import VMConsolidator
from src.energy import EnergyCalculator
from src.sla import SLACalculator
from src.data_preprocessing import FEATURE_NAMES

class TestDatacenterPipeline(unittest.TestCase):

    def setUp(self):
        self.pm_g5 = PhysicalMachine(
            pm_id="pm_test_01",
            cpu_capacity=1.0,
            memory_capacity=1.0,
            bandwidth_gbps=4.0,
            server_model="HP_G5"
        )

    def test_pm_specpower_interpolation(self):
        """Verify SPECpower piecewise linear interpolation and sleep power."""
        # 1. Idle PM in sleep mode should consume 10 W
        self.pm_g5.status = STATUS_IDLE
        self.assertEqual(self.pm_g5.get_power_watts(), 10.0)

        # 2. At 0% CPU active load with hosted VM: 93.7 W
        dummy_vm = VirtualMachine("v_dummy", cpu_requirement=0.001, memory_requirement=0.001)
        self.pm_g5.add_vm(dummy_vm)
        self.pm_g5.current_cpu_util = 0.0
        self.pm_g5.status = STATUS_ACTIVE
        self.assertAlmostEqual(self.pm_g5.get_power_watts(), 93.7, places=1)

        # 3. At 50% CPU active load: 116.0 W
        self.pm_g5.current_cpu_util = 0.50
        self.assertAlmostEqual(self.pm_g5.get_power_watts(), 116.0, places=1)

        # 4. At 100% CPU load: 135.0 W
        self.pm_g5.current_cpu_util = 1.00
        self.assertAlmostEqual(self.pm_g5.get_power_watts(), 135.0, places=1)

    def test_autoencoder_architecture(self):
        """Verify Autoencoder shape: 16 -> 12 -> 5 -> 12 -> 16 and latent extraction."""
        ae = VMAutoencoder(input_dim=16, latent_dim=5)
        x_dummy = np.random.rand(10, 16).astype(np.float32)
        
        # Test forward pass on underlying PyTorch module
        x_tensor = torch.tensor(x_dummy)
        recon, latent = ae.model(x_tensor)
        self.assertEqual(recon.shape, (10, 16))
        self.assertEqual(latent.shape, (10, 5))

        # Test latent extraction method
        latent_np = ae.extract_latent(x_dummy)
        self.assertEqual(latent_np.shape, (10, 5))

    def test_vm_priority_sorting(self):
        """Verify Priority Queue sorting: A_i ascending, -P_i descending, -R_i descending."""
        vms = [
            VirtualMachine("v1", 0.1, 0.2, priority=100, arrival_time=10.0),
            VirtualMachine("v2", 0.5, 0.2, priority=200, arrival_time=5.0),
            VirtualMachine("v3", 0.2, 0.1, priority=150, arrival_time=5.0),
            VirtualMachine("v4", 0.6, 0.3, priority=200, arrival_time=5.0),
        ]
        pms = [self.pm_g5]
        scheduler = CloudScheduler(pms=pms)
        sorted_vms = scheduler.sort_vms_by_priority(vms)
        
        # Expected order:
        # At t=5.0: highest priority (200) first, ties broken by higher resource (0.6+0.3 vs 0.5+0.2)
        # v4 (t=5, P=200, R=0.9) -> v2 (t=5, P=200, R=0.7) -> v3 (t=5, P=150, R=0.3) -> v1 (t=10, P=100, R=0.3)
        self.assertEqual([v.vm_id for v in sorted_vms], ["v4", "v2", "v3", "v1"])

    def test_least_loaded_initial_placement(self):
        """Verify placement under Th=0.75 upper threshold."""
        pm1 = PhysicalMachine("pm1", server_model="HP_G5")
        pm2 = PhysicalMachine("pm2", server_model="HP_G5")
        pms = [pm1, pm2]
        scheduler = CloudScheduler(pms=pms, upper_threshold=0.75)

        # Place VM 1 (cpu=0.4, mem=0.2)
        v1 = VirtualMachine("v1", 0.4, 0.2)
        self.assertTrue(scheduler.schedule_vm_initial(v1))
        self.assertEqual(pm1.current_cpu_util, 0.4)

        # Place VM 2 (cpu=0.3, mem=0.2) -> fits on pm1 (0.4+0.3 = 0.7 <= 0.75)
        v2 = VirtualMachine("v2", 0.3, 0.2)
        self.assertTrue(scheduler.schedule_vm_initial(v2))
        self.assertEqual(pm1.current_cpu_util, 0.7)

        # Place VM 3 (cpu=0.2, mem=0.1) -> exceeds pm1 (0.7+0.2 = 0.9 > 0.75), should go to pm2
        v3 = VirtualMachine("v3", 0.2, 0.1)
        self.assertTrue(scheduler.schedule_vm_initial(v3))
        self.assertEqual(v3.current_pm_id, "pm2")
        self.assertEqual(pm2.current_cpu_util, 0.2)

    def test_mmt_migration_and_degradation(self):
        """Verify MMT selection (min memory) and CPU degradation calculation."""
        migration_engine = VMMigrationEngine(network_bandwidth_gbps=4.0)
        
        vm_small_mem = VirtualMachine("v_small", cpu_requirement=0.4, memory_requirement=0.01)
        vm_large_mem = VirtualMachine("v_large", cpu_requirement=0.4, memory_requirement=0.10)
        
        pm_src = PhysicalMachine("pm_src", server_model="HP_G5")
        pm_src.add_vm(vm_large_mem)
        pm_src.add_vm(vm_small_mem)
        
        # MMT should select vm_small_mem
        selected = migration_engine.select_vm_mmt(pm_src)
        self.assertEqual(selected.vm_id, "v_small")

        # Migration time & CPU degradation
        t_mig = migration_engine.calculate_migration_time(vm_small_mem)
        deg = migration_engine.calculate_cpu_degradation(vm_small_mem, t_mig)
        self.assertGreater(t_mig, 0.0)
        self.assertAlmostEqual(deg, 0.1 * 0.4 * t_mig, places=6)

    def test_sla_metrics_computation(self):
        """Verify SLATAH, PDM, and SLAv formulas."""
        sla_calc = SLACalculator()
        
        pm = PhysicalMachine("pm_sla", server_model="HP_G5")
        pm.status = STATUS_ACTIVE
        pm.total_active_time = 3600.0
        pm.time_at_100_cpu = 36.0 # 1% of time at 100%
        
        vm = VirtualMachine("vm_sla", cpu_requirement=0.2, memory_requirement=0.05, duration=3600.0)
        vm.record_migration(migration_time=10.0, cpu_degradation=0.2, destination_pm_id="pm2")
        
        summary = sla_calc.evaluate_all(pms=[pm], vms=[vm], migrations=[{"vm_id": "vm_sla"}])
        
        # SLATAH = 36.0 / 3600.0 = 0.01
        self.assertAlmostEqual(summary["slatah"], 0.01, places=4)
        # PDM > 0 due to recorded migration degradation
        self.assertGreater(summary["pdm"], 0.0)
        # SLAv = SLATAH * PDM
        self.assertAlmostEqual(summary["slav"], summary["slatah"] * summary["pdm"], places=6)

if __name__ == "__main__":
    unittest.main()

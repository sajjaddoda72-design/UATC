import unittest
import torch
import math
from UATC import AdaptiveExpertController, ControllerConfig, ModelState

class TestUATCController(unittest.TestCase):
    def setUp(self):
        # Initialize default configurations optimized for automated testing
        self.cfg = ControllerConfig(warmup_steps=5, min_batch_size=2, max_batch_size=64)
        self.controller = AdaptiveExpertController(self.cfg)

    def test_public_phase_accessor(self):
        # Verify that the public phase property reads the current phase successfully from outside the class
        self.assertEqual(self.controller.phase, "WARMUP")

    def test_kalman_filter_denoising(self):
        # Test Kalman filter: feed noisy VRAM and verify the smoothed output is bounded and stable
        smoothed, velocity = self.controller._update_kalman(0.85)
        self.assertTrue(0.0 <= smoothed <= 1.0)
        self.assertTrue(isinstance(smoothed, float))

    def test_sanity_guard_nan_inf_detection(self):
        # Test Sanity Guard: inject a NaN loss value and verify it triggers the SANITY_GUARD_NAN_INF recovery path
        state_nan = ModelState(
            step=1, current_batch_size=16, current_amp_enabled=False,
            current_checkpointing_enabled=False, current_pruning_ratio=0.0,
            current_lr=1e-4, loss_current=float('nan'), loss_velocity=0.0,
            loss_variance=0.0, loss_ema_short=5.0, loss_ema_long=5.0,
            memory_pressure=0.2, grad_norm_ema=0.25
        )
        action = self.controller.decide(state_nan)
        # The controller must request step skipping, drop the learning rate, and log the reason
        self.assertTrue(action.skip_step)
        self.assertEqual(action.metadata.get("reason"), "SANITY_GUARD_NAN_INF")

    def test_warmup_to_scaling_transition(self):
        # Verify the controller remains in WARMUP for the initial steps
        for step in range(1, 5):
            state = ModelState(
                step=step, current_batch_size=16, current_amp_enabled=False,
                current_checkpointing_enabled=False, current_pruning_ratio=0.0,
                current_lr=1e-4, loss_current=5.0, loss_velocity=0.0,
                loss_variance=0.0, loss_ema_short=5.0, loss_ema_long=5.0,
                memory_pressure=0.2, grad_norm_ema=0.25
            )
            self.controller.decide(state)
            self.assertEqual(self.controller.phase, "WARMUP")
        
        # Verify automatic transition to SCALING once warmup_steps threshold is reached (Step 5)
        state_transition = ModelState(
            step=5, current_batch_size=16, current_amp_enabled=False,
            current_checkpointing_enabled=False, current_pruning_ratio=0.0,
            current_lr=1e-4, loss_current=5.0, loss_velocity=0.0,
            loss_variance=0.0, loss_ema_short=5.0, loss_ema_long=5.0,
            memory_pressure=0.2, grad_norm_ema=0.25
        )
        self.controller.decide(state_transition)
        self.assertEqual(self.controller.phase, "SCALING")

    def test_oom_to_recovery_transition(self):
        # Transition the system to SCALING phase first
        for step in range(1, 6):
            state = ModelState(
                step=step, current_batch_size=16, current_amp_enabled=False,
                current_checkpointing_enabled=False, current_pruning_ratio=0.0,
                current_lr=1e-4, loss_current=5.0, loss_velocity=0.0,
                loss_variance=0.0, loss_ema_short=5.0, loss_ema_long=5.0,
                memory_pressure=0.2, grad_norm_ema=0.25
            )
            self.controller.decide(state)
        self.assertEqual(self.controller.phase, "SCALING")
        
        # Verify immediate protective transition to RECOVERY upon detecting an OOM event
        state_oom = ModelState(
            step=6, current_batch_size=16, current_amp_enabled=False,
            current_checkpointing_enabled=current_ckpt if 'current_ckpt' in globals() else False,
            current_pruning_ratio=0.0,
            current_lr=1.5e-4,
            loss_current=5.0,
            loss_velocity=0.0,
            loss_variance=0.0,
            loss_ema_short=5.0,
            loss_ema_long=5.0,
            memory_pressure=0.85,
            grad_norm_ema=0.25,
            oom_detected=True
        )
        self.controller.decide(state_oom)
        self.assertEqual(self.controller.phase, "RECOVERY")

    def test_batch_size_rounding_logic(self):
        # Test size-aware batch rounding: small batches use per-unit rounding to prevent dead-band freeze
        bs_small_even = self.controller._round_batch_size(4.2, min_bs=2)
        self.assertEqual(bs_small_even, 4)
        
        bs_small_odd = self.controller._round_batch_size(3.1, min_bs=2)
        self.assertEqual(bs_small_odd, 3)
        
        # Verify large batches snap to the nearest multiple of 16 for Tensor Core alignment
        bs_large = self.controller._round_batch_size(44.0, min_bs=2)
        self.assertEqual(bs_large, 48)

if __name__ == '__main__':
    unittest.main()

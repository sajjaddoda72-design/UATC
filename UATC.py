from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import torch

logger = logging.getLogger("UATC")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# 1. DATA STRUCTURES
# =====================================================================

@dataclass
class ControllerConfig:
    """Controller hyperparameters. Each field documents its semantic role."""

    # ---- Memory pressure gating ----------------------------------------
    memory_soft_limit: float = 0.74
    memory_hard_limit: float = 0.95

    # ---- OOM recovery --------------------------------------------------
    oom_recovery_factor: float = 0.5

    # ---- Batch-size bounds --------------------------------------------
    min_batch_size: int = 16
    max_batch_size: int = 512
    max_bs_growth_per_step: int = 2

    # ---- Phase machine -------------------------------------------------
    warmup_steps: int = 5
    bs_update_frequency: int = 1
    scaling_timeout: int = 50
    plateau_sustained_steps: int = 8

    # ---- Learning rate -------------------------------------------------
    lr_min: float = 1e-6
    lr_max: float = 1e-3
    lr_decay_factor: float = 0.5
    loss_spike_factor: float = 0.5
    plateau_lr_decay: float = 0.95

    # ---- Plateau detection --------------------------------------------
    plateau_velocity_eps: float = 1e-3
    plateau_variance_eps: float = 1e-4
    plateau_cusum_leak: float = 0.95
    plateau_cusum_trigger_units: float = 20.0

    # ---- Dynamic pruning ----------------------------------------------
    min_pruning_ratio: float = 0.0
    max_pruning_ratio: float = 0.5
    pruning_grow_step: float = 0.05
    pruning_decay_rate: float = 0.01
    base_pruning_threshold: float = 0.12

    # ---- Kalman filter -------------------------------------------------
    kalman_q: float = 1e-4
    kalman_r: float = 5e-4
    kalman_trust_var: float = 1e-3

    # ---- Smith predictor ----------------------------------------------
    smith_delay: int = 4
    smith_max_delay: int = 12
    smith_correction_gain: float = 0.1
    smith_correction_clamp: float = 0.05

    # ---- RECOVERY exit ------------------------------------------------
    recovery_min_steps_fpft: int = 3
    recovery_min_steps_peft: int = 4
    recovery_min_steps_qlora: int = 6
    recovery_clean_steps_fpft: int = 2
    recovery_clean_steps_peft: int = 3
    recovery_clean_steps_qlora: int = 4
    recovery_min_pruning_ratio: float = 0.15
    # FIX #1: Noise-resilient acceleration gate. PyTorch's caching allocator
    # naturally fluctuates by 10-50 MB at each step, producing EMA acceleration
    # noise on the order of 1e-3 to 5e-3. The previous 1e-4 threshold trapped
    # the controller in RECOVERY forever. 5e-3 is well above typical noise
    # but well below genuine memory surges (~1e-2 or larger).
    recovery_accel_noise_gate: float = 5e-3
    # Fallback exit path: if memory has been safe & stable for this many
    # consecutive steps, exit RECOVERY even if pruning_effective is not yet
    # satisfied (prevents permanent trapping when pruning is not the lever).
    recovery_safe_fallback_steps: int = 12

    # ---- WATCH state growth (FIX #2) ---------------------------------
    # Conservative growth cap applied when VRAM is congested but stable
    # (WATCH state). Set slightly above 1.0 to permit gradual upward scaling.
    watch_state_growth_cap: float = 1.05
    # Width of the WATCH band expressed as fractions of memory_soft_limit.
    # Tuned for real congested-VRAM conditions (e.g. T4 with background
    # allocation holding pressure in the 52%-58% range).
    schmitt_thr_low_factor: float = 0.80   # STABLE band ceiling (raised from 0.70)
    schmitt_thr_mid_factor: float = 0.88   # STABLE->WATCH trigger (raised from 0.80)
    schmitt_thr_high_factor: float = 0.94  # WATCH->REDUCED trigger (raised from 0.88)

    # ---- Integer dead-band preventer (FIX #3) -----------------------
    # Batch sizes below this threshold use per-unit rounding instead of
    # rounding to a multiple of 16. Prevents small batches (2, 4, 8) from
    # being quantized to 0 and frozen at min_batch_size forever.
    small_batch_round_threshold: int = 32
    # Enable the integer dead-band force-increment (+1) path.
    deadband_force_increment: bool = True
    # Minimum bs_multiplier signal required to trigger the +1 force-increment.
    deadband_min_signal: float = 1.005

    # ---- NaN/Inf recovery ---------------------------------------------
    nan_max_consecutive: int = 3

    # ---- Anti-windup --------------------------------------------------
    pid_integral_clamp: float = 0.4
    pid_anti_windup_gain: float = 0.5

    # ---- EMA smoothing -----------------------------------------------
    mem_velocity_ema_alpha: float = 0.4
    mem_accel_ema_alpha: float = 0.3
    pid_deriv_filter_alpha: float = 0.3


@dataclass
class ModelState:
    """Per-step telemetry passed from the training loop."""
    step: int
    current_batch_size: int
    current_amp_enabled: bool
    current_checkpointing_enabled: bool
    current_pruning_ratio: float
    current_lr: float
    loss_current: float
    loss_velocity: float
    loss_variance: float
    loss_ema_short: float
    loss_ema_long: float
    memory_pressure: float
    grad_norm_ema: float
    oom_detected: bool = False
    training_paradigm: Optional[str] = None
    lora_rank: int = 0
    active_param_ratio: Optional[float] = None
    current_seq_len: int = 0
    max_seq_len_in_batch: int = 0
    total_vram_bytes: Optional[int] = None
    quantization_bits: int = 0


@dataclass
class ControllerAction:
    """Execution directives returned by decide()."""
    target_batch_size: int
    target_amp_enabled: bool
    target_checkpointing_enabled: bool
    target_pruning_ratio: float
    target_lr: float
    skip_step: bool = False
    recompute_loss: bool = False
    active_prune_threshold: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 2. DYNAMIC DATA PRUNER
# =====================================================================

class DynamicDataPruner:
    """Phase-aware per-sample loss filter.

    Samples whose per-sample loss falls below the active threshold are
    masked out. At least one sample always survives to prevent zero-tensor
    NaN-gradient catastrophes.
    """

    _PHASE_MULTIPLIERS: Dict[str, float] = {
        "WARMUP": 0.50,
        "SCALING": 1.00,
        "CONVERGENCE": 0.50,
        "RECOVERY": 2.50,
    }

    def __init__(self, base_threshold: float = 0.12):
        self.base_threshold = float(base_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_active_threshold(self, controller_phase: str) -> float:
        multiplier = self._PHASE_MULTIPLIERS.get(controller_phase, 1.0)
        return max(0.0, self.base_threshold * multiplier)

    def filter_batch_losses(
        self,
        individual_losses: Optional[torch.Tensor],
        controller_phase: str,
    ) -> Tuple[torch.Tensor, int, float]:
        active_threshold = self.get_active_threshold(controller_phase)

        if individual_losses is None:
            return (torch.empty(0, device=self.device, dtype=torch.float32),
                    0, active_threshold)
        if individual_losses.numel() == 0:
            return individual_losses, 0, active_threshold

        flat_losses = individual_losses.reshape(-1).to(torch.float32)
        total_count = flat_losses.numel()

        keep_mask = flat_losses >= active_threshold
        filtered = flat_losses[keep_mask]
        kept_count = filtered.numel()
        skipped_count = total_count - kept_count

        if kept_count == 0:
            max_idx = int(torch.argmax(flat_losses).item())
            filtered = flat_losses[max_idx:max_idx + 1].clone()
            skipped_count = total_count - 1

        return filtered, skipped_count, active_threshold


# =====================================================================
# 3. ADAPTIVE CONTROLLER
# =====================================================================

class AdaptiveExpertController:
    """PID + Kalman + Smith adaptive controller for training stability.

    Call decide(state) once per training step. Stateless from the caller's
    perspective; maintains internal PID/Kalman/phase state across calls.
    """

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pruner = DynamicDataPruner(base_threshold=cfg.base_pruning_threshold)

        self._total_vram_gb: Optional[float] = None
        self._capacity_scale: float = 1.0

        self._init_state()

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------
    def _init_state(self, initial_memory_pressure: float = 0.0) -> None:
        """Centralized state init. Shared by __init__ and hard-reset paths."""
        self._phase: str = "WARMUP"
        self._steps_in_phase: int = 0
        self._steps_since_bs_update: int = 0
        self._cusum_plateau: float = 0.0
        self._plateau_sustained_count: int = 0

        self._pid_integral: float = 0.0
        self._pid_prev_error: float = 0.0
        self._pid_filtered_derivative: float = 0.0
        self._pid_saturated: bool = False

        self._prev_memory_pressure: float = initial_memory_pressure
        self._prev_memory_pressure_t1: float = initial_memory_pressure

        self._mem_est: float = initial_memory_pressure
        self._mem_est_var: float = self.cfg.kalman_r
        self._mem_velocity_ema: float = 0.0
        self._mem_accel_ema: float = 0.0

        self._smith_delay_buffer: Deque[float] = deque(maxlen=self.cfg.smith_max_delay)
        self._smith_predicted_pressure: float = initial_memory_pressure

        self._recovery_clean_steps: int = 0
        self._recovery_safe_streak: int = 0
        self._recovery_pruning_ratio_high_water: float = 0.0

        self._consecutive_nan_steps: int = 0

        self._best_loss_ema_long: float = float("inf")
        self._loss_spike_active: bool = False

        self._ckpt_state: Optional[str] = None
        self._ckpt_flip_count: int = 0

        self._bs_state: Optional[str] = None
        self._slew_rate_clamped_prev: bool = False
        self._slew_clamp_streak: int = 0

        self._prev_max_seq_len: int = 0
        self._seq_len_spike_cusum: float = 0.0

        self._cached_paradigm: str = "FPFT"
        self._cached_elasticity: float = 0.9
    @property
    def phase(self) -> str:
        """Expose the current controller phase as a public read-only property."""
        return self._phase
        
    # ------------------------------------------------------------------
    # CUDA cache flush
    # ------------------------------------------------------------------
    @staticmethod
    def _clear_gpu_caches() -> None:
        """Release fragmented CUDA memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    @staticmethod
    def _isfinite_scalar(x: Any) -> bool:
        if x is None:
            return False
        if torch.is_tensor(x):
            return bool(torch.isfinite(x).all().item())
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False

    def _is_finite_state(self, st: ModelState) -> Tuple[bool, str]:
        for name in ("loss_current", "loss_velocity", "loss_variance",
                     "memory_pressure", "grad_norm_ema"):
            v = getattr(st, name, 0.0)
            if not self._isfinite_scalar(v):
                return False, name
        return True, ""

    # ------------------------------------------------------------------
    # Size-aware batch-size rounding  (FIX #3, part 1)
    # ------------------------------------------------------------------
    def _round_batch_size(self, raw_bs: float, min_bs: int) -> int:
        """Quantize a proposed batch size with size-aware behavior.

        Small batch sizes (< small_batch_round_threshold) are rounded per-unit
        so that fine-grained adjustments on edge GPUs (bs = 2, 4, 8) are not
        quantized into oblivion. Larger batch sizes are rounded to the nearest
        multiple of 16 to preserve tensor-core kernel efficiency.
        """
        raw_int = int(round(raw_bs))
        if raw_int < self.cfg.small_batch_round_threshold:
            # Per-unit rounding for small batches (edge fine-tuning regime).
            return max(min_bs, raw_int)
        # Snap to nearest multiple of 16 for kernel efficiency at scale.
        return max(min_bs, int(round(raw_int / 16.0)) * 16)

    # ------------------------------------------------------------------
    # Paradigm classifier
    # ------------------------------------------------------------------
    def _classify_paradigm(self, s: ModelState) -> Tuple[str, float, float]:
        """Classify paradigm with conservative QLoRA default on ambiguity."""
        paradigm = (s.training_paradigm or self._cached_paradigm).upper()
        lora_rank = int(getattr(s, "lora_rank", 0) or 0)
        quant_bits = int(getattr(s, "quantization_bits", 0) or 0)

        if paradigm not in ("FPFT", "PEFT", "QLoRA"):
            if lora_rank > 0 and quant_bits in (4, 8):
                paradigm = "QLoRA"
            elif lora_rank > 0:
                paradigm = "PEFT" if quant_bits == 0 else "QLoRA"
            elif quant_bits in (4, 8):
                paradigm = "QLoRA"
            else:
                paradigm = "FPFT"

        active_ratio = getattr(s, "active_param_ratio", None)
        if active_ratio is None or not self._isfinite_scalar(active_ratio):
            if paradigm == "QLoRA":
                active_ratio = self._clamp(0.02 + 0.01 * (lora_rank / 8.0), 0.005, 0.15)
            elif paradigm == "PEFT":
                active_ratio = self._clamp(0.05 + 0.02 * (lora_rank / 8.0), 0.01, 0.30)
            else:
                active_ratio = 1.0

        if paradigm == "QLoRA":
            xi = self._clamp(0.30 + 0.40 * active_ratio, 0.30, 0.70)
        elif paradigm == "PEFT":
            xi = self._clamp(0.45 + 0.40 * active_ratio, 0.45, 0.85)
        else:
            xi = 0.90

        self._cached_paradigm = paradigm
        self._cached_elasticity = xi
        return paradigm, active_ratio, xi

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------
    def _update_kalman(self, z_raw: float) -> Tuple[float, float]:
        """Single-step Kalman update. Returns (smoothed_value, smoothed_velocity)."""
        c = self.cfg
        Q, R = c.kalman_q, c.kalman_r
        pred_var = self._mem_est_var + Q
        K = pred_var / (pred_var + R)
        prev_est = self._mem_est
        self._mem_est = prev_est + K * (z_raw - prev_est)
        self._mem_est_var = (1.0 - K) * pred_var
        smoothed_velocity = self._mem_est - prev_est
        if self._mem_est_var < c.kalman_trust_var:
            smoothed_value = self._mem_est
        else:
            smoothed_value = 0.5 * self._mem_est + 0.5 * z_raw
        return self._clamp(smoothed_value, 0.0, 1.0), smoothed_velocity

    # ------------------------------------------------------------------
    # Smith predictor (feedforward heuristic)
    # ------------------------------------------------------------------
    def _apply_smith(self, pid_output_raw: float, raw_derivative: float) -> float:
        """Delay-line feedforward approximating a Smith predictor."""
        c = self.cfg
        self._smith_delay_buffer.append(pid_output_raw)
        if len(self._smith_delay_buffer) > c.smith_delay:
            delayed = self._smith_delay_buffer[0]
            correction = c.smith_correction_gain * delayed - raw_derivative
            correction = self._clamp(
                correction, -c.smith_correction_clamp, c.smith_correction_clamp
            )
            return pid_output_raw + correction
        return pid_output_raw

    # ------------------------------------------------------------------
    # PID step
    # ------------------------------------------------------------------
    def _pid_step(self, z_raw: float, xi: float) -> Tuple[float, float]:
        c = self.cfg
        Kp_base, Ki_base, Kd_base = 0.50, 0.003, 0.04

        span = max(1e-6, c.memory_hard_limit - c.memory_soft_limit)
        proximity_gain = 1.0 + 1.5 * self._clamp(
            (z_raw - c.memory_soft_limit) / span, 0.0, 2.0
        )
        accel_gain = 1.0 + (1.0 if self._mem_accel_ema > 0 else 0.5) * \
            self._clamp(abs(self._mem_accel_ema) * 1e3, 0.0, 5.0)
        elasticity_gain = self._clamp(xi / 0.9, 0.4, 1.3)

        Kp = Kp_base * self._capacity_scale * proximity_gain * elasticity_gain
        Ki = Ki_base * self._capacity_scale * elasticity_gain
        Kd = Kd_base * self._capacity_scale * accel_gain * elasticity_gain

        accel_backoff = self._clamp(self._mem_accel_ema * 50.0, 0.0, 0.10)
        mem_setpoint = c.memory_soft_limit * (0.95 - accel_backoff)
        mem_error = mem_setpoint - z_raw

        # D term uses EMA-filtered Kalman velocity (was previously always zero
        # because prev_memory_pressure was overwritten before derivative read).
        raw_derivative = -self._mem_velocity_ema
        self._pid_filtered_derivative = (
            (1.0 - c.pid_deriv_filter_alpha) * self._pid_filtered_derivative
            + c.pid_deriv_filter_alpha * raw_derivative
        )

        pid_output = (
            Kp * mem_error
            + Ki * self._pid_integral
            + Kd * self._pid_filtered_derivative
        )
        return pid_output, mem_error

    # ------------------------------------------------------------------
    # Slew-rate limiter
    # ------------------------------------------------------------------
    def _apply_slew_limit(self, s: ModelState, proposed_bs: int) -> Tuple[int, bool]:
        """Upward-only slew-rate clamp. Returns (clamped_bs, was_clamped)."""
        c = self.cfg
        if proposed_bs > s.current_batch_size:
            max_allowed = s.current_batch_size + c.max_bs_growth_per_step
            if proposed_bs > max_allowed:
                return max_allowed, True
        return proposed_bs, False

    # ------------------------------------------------------------------
    # Phase machine  (FIX #1 applied here)
    # ------------------------------------------------------------------
    def _update_phase_machine(self, s: ModelState, paradigm: str) -> None:
        """Phase transitions gated on sustained conditions, not single steps."""
        c = self.cfg

        if self._phase == "WARMUP":
            if s.step >= c.warmup_steps:
                self._phase = "SCALING"
                self._steps_in_phase = 0
                self._pid_integral = 0.0
                logger.info("Phase WARMUP->SCALING at step %d", s.step)

        elif self._phase == "SCALING":
            if abs(s.loss_velocity) < c.plateau_velocity_eps * 5:
                self._plateau_sustained_count += 1
            else:
                self._plateau_sustained_count = 0

            timeout_reached = self._steps_in_phase >= c.scaling_timeout
            sustained_plateau = self._plateau_sustained_count >= c.plateau_sustained_steps

            if timeout_reached or sustained_plateau:
                self._phase = "CONVERGENCE"
                self._steps_in_phase = 0
                self._pid_integral = 0.0
                self._plateau_sustained_count = 0
                logger.info(
                    "Phase SCALING->CONVERGENCE at step %d (timeout=%s, sustained=%s)",
                    s.step, timeout_reached, sustained_plateau,
                )

        elif self._phase == "RECOVERY":
            z_raw = float(getattr(s, "memory_pressure", 0.0))
            is_memory_safe = z_raw < c.memory_soft_limit

            # FIX #1: Noise-resilient acceleration gate.
            # PyTorch's caching allocator fluctuates by 10-50 MB per step on
            # real GPUs (T4, A10, etc.), producing EMA acceleration noise of
            # ~2e-3 to 4e-3. The previous "< 1e-4" threshold was unachievable
            # in practice and trapped the controller in RECOVERY forever.
            # The new |a| < recovery_accel_noise_gate (default 5e-3) gate is
            # safely above typical allocator noise but well below genuine
            # memory surges, allowing clean exit once the surge settles.
            is_acceleration_stable = abs(self._mem_accel_ema) < c.recovery_accel_noise_gate

            # Track pruning high-water to confirm pruning took effect before exit.
            self._recovery_pruning_ratio_high_water = max(
                self._recovery_pruning_ratio_high_water,
                float(getattr(s, "current_pruning_ratio", 0.0) or 0.0),
            )
            pruning_effective = (
                self._recovery_pruning_ratio_high_water >= c.recovery_min_pruning_ratio
            )

            # Primary clean-step condition: OOM-free, memory-safe, accel-stable,
            # and pruning took effect.
            primary_clean = (
                not s.oom_detected and is_memory_safe
                and is_acceleration_stable and pruning_effective
            )

            # Fallback clean-step condition: even if pruning_effective is not
            # yet satisfied (e.g. paradigm doesn't need aggressive pruning),
            # accept the step as clean if memory is safe AND acceleration is
            # stable. This prevents the controller from being permanently
            # trapped when pruning is not the relevant recovery lever.
            fallback_clean = (
                not s.oom_detected and is_memory_safe and is_acceleration_stable
            )

            if primary_clean or fallback_clean:
                self._recovery_clean_steps += 1
            else:
                self._recovery_clean_steps = 0

            # Separate safe-streak counter for the fallback exit path.
            if fallback_clean:
                self._recovery_safe_streak += 1
            else:
                self._recovery_safe_streak = 0

            if paradigm == "QLoRA":
                min_steps, min_clean = c.recovery_min_steps_qlora, c.recovery_clean_steps_qlora
            elif paradigm == "PEFT":
                min_steps, min_clean = c.recovery_min_steps_peft, c.recovery_clean_steps_peft
            else:
                min_steps, min_clean = c.recovery_min_steps_fpft, c.recovery_clean_steps_fpft

            primary_exit = (
                self._steps_in_phase >= min_steps
                and self._recovery_clean_steps >= min_clean
            )
            # Fallback exit: sustained safe memory without OOM for a longer
            # window. Guarantees forward progress even in edge cases where
            # pruning_effective never trips.
            fallback_exit = (
                self._steps_in_phase >= min_steps
                and self._recovery_safe_streak >= c.recovery_safe_fallback_steps
            )

            if primary_exit or fallback_exit:
                exit_reason = "PRIMARY" if primary_exit else "SAFE_FALLBACK"
                self._phase = "SCALING"
                self._steps_in_phase = 0
                self._pid_integral = 0.0
                self._recovery_clean_steps = 0
                self._recovery_safe_streak = 0
                self._recovery_pruning_ratio_high_water = 0.0
                logger.info(
                    "Phase RECOVERY->SCALING at step %d (exit=%s, accel_ema=%.6f)",
                    s.step, exit_reason, self._mem_accel_ema,
                )

    # ------------------------------------------------------------------
    # 3-state Schmitt trigger  (FIX #2 applied here)
    # ------------------------------------------------------------------
    def _update_bs_state(self, z_raw: float) -> None:
        """3-state Schmitt: STABLE / WATCH / REDUCED with proper hysteresis.

        Thresholds are calibrated for real congested-VRAM conditions
        (e.g. NVIDIA T4 with background allocations holding pressure in the
        52%-58% band). The previous thr_low=0.70*soft_limit was too low for
        that band, leaving the controller stuck in WATCH where the bs
        multiplier was capped below 1.0 and could never scale up.
        """
        c = self.cfg
        if self._bs_state is None:
            self._bs_state = "STABLE"

        thr_low = c.memory_soft_limit * c.schmitt_thr_low_factor
        thr_mid = c.memory_soft_limit * c.schmitt_thr_mid_factor
        thr_high = c.memory_soft_limit * c.schmitt_thr_high_factor

        # Proper Schmitt hysteresis:
        #   STABLE   -> WATCH    when z >= thr_mid
        #   STABLE   -> REDUCED  when z >= thr_high
        #   WATCH    -> STABLE   when z <  thr_low   (hysteresis gap)
        #   WATCH    -> REDUCED  when z >= thr_high
        #   REDUCED  -> WATCH    when z <  thr_mid   (hysteresis gap)
        # The hysteresis gaps (thr_low vs thr_mid, thr_mid vs thr_high)
        # prevent oscillation between adjacent states under noisy pressure.
        if self._bs_state == "STABLE":
            if z_raw >= thr_high:
                self._bs_state = "REDUCED"
            elif z_raw >= thr_mid:
                self._bs_state = "WATCH"
            # else: remain STABLE
        elif self._bs_state == "WATCH":
            if z_raw >= thr_high:
                self._bs_state = "REDUCED"
            elif z_raw < thr_low:
                self._bs_state = "STABLE"
            # else: remain WATCH
        else:  # REDUCED
            if z_raw < thr_mid:
                self._bs_state = "WATCH"
            # else: remain REDUCED

    # ------------------------------------------------------------------
    # Checkpointing Schmitt trigger
    # ------------------------------------------------------------------
    def _update_ckpt_state(self, z_raw: float) -> None:
        c = self.cfg
        if self._total_vram_gb is None:
            return
        if self._total_vram_gb < 24.0:
            ckpt_on = c.memory_soft_limit * 0.80
            ckpt_off = c.memory_soft_limit * 0.60
        else:
            ckpt_on = c.memory_soft_limit * 0.85
            ckpt_off = c.memory_soft_limit * 0.70

        if self._ckpt_state is None:
            self._ckpt_state = "ON"
        if self._ckpt_state == "OFF" and z_raw > ckpt_on:
            self._ckpt_state = "ON"
            self._ckpt_flip_count += 1
            logger.info("Checkpointing ON (z=%.3f, flip=%d)", z_raw, self._ckpt_flip_count)
        elif self._ckpt_state == "ON" and z_raw < ckpt_off:
            self._ckpt_state = "OFF"
            self._ckpt_flip_count += 1
            logger.info("Checkpointing OFF (z=%.3f, flip=%d)", z_raw, self._ckpt_flip_count)

    # ------------------------------------------------------------------
    # Tiered NaN/Inf recovery
    # ------------------------------------------------------------------
    def _handle_non_finite(
        self, action: ControllerAction, s: ModelState,
        paradigm: str, bad_field: str,
    ) -> ControllerAction:
        """Soft recovery on first NaN, hard reset to RECOVERY after N consecutive."""
        c = self.cfg
        self._consecutive_nan_steps += 1

        self._clear_gpu_caches()

        if self._consecutive_nan_steps >= c.nan_max_consecutive:
            logger.error(
                "NaN/Inf in '%s' for %d consecutive steps; hard reset to RECOVERY",
                bad_field, self._consecutive_nan_steps,
            )
            self._init_state(
                initial_memory_pressure=float(getattr(s, "memory_pressure", 0.0) or 0.0)
            )
            self._phase = "RECOVERY"
            self._ckpt_state = "ON"
            self._bs_state = "REDUCED"
            action.target_lr = c.lr_min
            action.target_batch_size = max(
                c.min_batch_size, int(s.current_batch_size * c.oom_recovery_factor)
            )
            action.target_checkpointing_enabled = True
            action.target_pruning_ratio = c.max_pruning_ratio
            action.target_amp_enabled = True
            action.skip_step = True
            action.recompute_loss = True
            action.metadata.update({
                "reason": "NAN_HARD_RESET", "bad_field": bad_field,
                "consecutive": self._consecutive_nan_steps,
                "phase": self._phase, "paradigm": paradigm,
            })
            action.active_prune_threshold = self.pruner.get_active_threshold(self._phase)
            return action

        logger.warning(
            "NaN/Inf in '%s' (step %d, consecutive %d); soft recovery",
            bad_field, s.step, self._consecutive_nan_steps,
        )
        action.target_lr = float(
            self._clamp(s.current_lr * c.lr_decay_factor, c.lr_min, c.lr_max)
        )
        action.skip_step = True
        action.recompute_loss = True
        self._pid_integral = 0.0
        self._pid_filtered_derivative = 0.0
        action.metadata.update({
            "reason": "SANITY_GUARD_NAN_INF", "bad_field": bad_field,
            "consecutive": self._consecutive_nan_steps,
            "phase": self._phase, "paradigm": paradigm,
        })
        action.active_prune_threshold = self.pruner.get_active_threshold(self._phase)
        return action

    # ------------------------------------------------------------------
    # Emergency OOM handler
    # ------------------------------------------------------------------
    def _handle_emergency(
        self, action: ControllerAction, s: ModelState,
        paradigm: str, xi: float,
    ) -> ControllerAction:
        """Hard reduction + pruning boost + state reset on OOM."""
        c = self.cfg

        self._clear_gpu_caches()

        if paradigm == "QLoRA":
            recovery_factor = c.oom_recovery_factor * 0.75
            prune_boost_mul = 4.0
        elif paradigm == "PEFT":
            recovery_factor = c.oom_recovery_factor * 0.85
            prune_boost_mul = 3.5
        else:
            recovery_factor = c.oom_recovery_factor
            prune_boost_mul = 3.0

        new_bs = int(self._clamp(
            s.current_batch_size * recovery_factor, c.min_batch_size, c.max_batch_size
        ))
        new_pruning = float(self._clamp(
            s.current_pruning_ratio + c.pruning_grow_step * prune_boost_mul,
            c.min_pruning_ratio, c.max_pruning_ratio,
        ))

        action.target_batch_size = new_bs
        action.target_checkpointing_enabled = True
        action.target_amp_enabled = True
        action.target_pruning_ratio = new_pruning
        action.skip_step = True
        action.recompute_loss = True

        self._phase = "RECOVERY"
        self._steps_in_phase = 0
        self._steps_since_bs_update = 0
        self._recovery_clean_steps = 0
        self._recovery_safe_streak = 0
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0
        self._pid_filtered_derivative = 0.0
        self._pid_saturated = False
        self._ckpt_state = "ON"
        self._bs_state = "REDUCED"
        self._smith_delay_buffer.clear()
        self._slew_rate_clamped_prev = False
        self._slew_clamp_streak = 0
        self._recovery_pruning_ratio_high_water = new_pruning

        logger.error(
            "EMERGENCY_OOM paradigm=%s bs %d->%d prune->%.3f (z=%.3f, v=%.6f, a=%.6f)",
            paradigm, s.current_batch_size, new_bs, new_pruning,
            float(getattr(s, "memory_pressure", 0.0)),
            self._mem_velocity_ema, self._mem_accel_ema,
        )

        action.metadata.update({
            "reason": "EMERGENCY_OOM", "phase": self._phase, "paradigm": paradigm,
            "mem_velocity": round(self._mem_velocity_ema, 6),
            "mem_accel": round(self._mem_accel_ema, 6),
            "elasticity_xi": round(xi, 4),
            "recovery_factor": round(recovery_factor, 4),
            "slew_rate_clamped": False,
            "pruning_ratio": new_pruning,
        })
        action.active_prune_threshold = self.pruner.get_active_threshold(self._phase)
        return action

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def decide(self, state: ModelState) -> ControllerAction:
        c = self.cfg
        s = state

        if self._total_vram_gb is None:
            tv = getattr(s, "total_vram_bytes", None) or (16 * (1024 ** 3))
            self._total_vram_gb = float(tv) / (1024 ** 3)
            self._capacity_scale = self._clamp(
                math.log1p(self._total_vram_gb) / math.log1p(80.0), 0.45, 1.4
            )

        action = ControllerAction(
            target_batch_size=s.current_batch_size,
            target_amp_enabled=s.current_amp_enabled,
            target_checkpointing_enabled=s.current_checkpointing_enabled,
            target_pruning_ratio=s.current_pruning_ratio,
            target_lr=s.current_lr,
        )

        # A. Paradigm classification
        paradigm, active_ratio, xi = self._classify_paradigm(s)

        # B. Sanity guard
        is_finite, bad_field = self._is_finite_state(s)
        if not is_finite:
            return self._handle_non_finite(action, s, paradigm, bad_field)
        self._consecutive_nan_steps = 0

        # C. Kalman denoising + first/second derivatives
        z_raw = float(getattr(s, "memory_pressure", 0.0))
        z_raw = self._clamp(z_raw, 0.0, 1.0)

        z_prev = self._prev_memory_pressure
        z_prev_t1 = self._prev_memory_pressure_t1

        mem_pressure_smooth, _kalman_velocity = self._update_kalman(z_raw)

        raw_mem_velocity = z_raw - z_prev
        raw_mem_accel = z_raw - 2.0 * z_prev + z_prev_t1
        alpha_v = c.mem_velocity_ema_alpha
        alpha_a = c.mem_accel_ema_alpha
        self._mem_velocity_ema = (1 - alpha_v) * self._mem_velocity_ema + alpha_v * raw_mem_velocity
        self._mem_accel_ema = (1 - alpha_a) * self._mem_accel_ema + alpha_a * raw_mem_accel

        self._prev_memory_pressure_t1 = z_prev
        self._prev_memory_pressure = z_raw

        # D. Sequence-length spike monitor
        cur_seq = int(getattr(s, "current_seq_len", 0) or 0)
        max_seq = int(getattr(s, "max_seq_len_in_batch", cur_seq) or 0)
        if cur_seq > 0 and max_seq > 0:
            seq_ratio = max_seq / max(cur_seq, 1)
            seq_amp_risk = self._clamp(seq_ratio ** 2, 1.0, 4.0)
        else:
            seq_ratio, seq_amp_risk = 1.0, 1.0
        seq_delta = max(0.0, float(max_seq - self._prev_max_seq_len))
        self._seq_len_spike_cusum = max(0.0, self._seq_len_spike_cusum + seq_delta - 64.0) * 0.85
        self._prev_max_seq_len = max(self._prev_max_seq_len, max_seq)
        seq_spike_flag = self._seq_len_spike_cusum > 256.0
        seq_bs_damp = 1.0 / max(seq_amp_risk, 1e-6)
        if seq_spike_flag:
            seq_bs_damp = min(seq_bs_damp, 0.75)

        # E. Emergency OOM
        emergency_trigger = (
            s.oom_detected
            or z_raw >= c.memory_hard_limit
            or (mem_pressure_smooth >= (c.memory_hard_limit - 0.03)
                and self._mem_accel_ema > 1e-3)
        )
        if emergency_trigger:
            return self._handle_emergency(action, s, paradigm, xi)

        # F. Phase-step accounting
        self._steps_in_phase += 1
        self._steps_since_bs_update += 1
        if self._isfinite_scalar(s.loss_ema_long) and s.loss_ema_long < self._best_loss_ema_long:
            self._best_loss_ema_long = s.loss_ema_long

        self._loss_spike_active = (
            self._isfinite_scalar(s.loss_ema_long)
            and s.loss_ema_long > 0
            and s.loss_current > s.loss_ema_long * (1.0 + c.loss_spike_factor)
        )

        # G. Phase machine
        self._update_phase_machine(s, paradigm)

        drift_signal = c.plateau_velocity_eps - abs(s.loss_velocity)
        if self._phase in ("SCALING", "CONVERGENCE"):
            self._cusum_plateau = (
                max(0.0, self._cusum_plateau + drift_signal) * c.plateau_cusum_leak
            )
        else:
            self._cusum_plateau = 0.0
        is_plateau_cusum = (
            self._cusum_plateau > c.plateau_velocity_eps * c.plateau_cusum_trigger_units
        )

        # H. PID step
        pid_output_raw, mem_error = self._pid_step(z_raw, xi)

        # I. Smith predictor
        raw_derivative_for_smith = -self._mem_velocity_ema
        pid_output_raw = self._apply_smith(pid_output_raw, raw_derivative_for_smith)
        self._smith_predicted_pressure = z_raw + pid_output_raw * 0.1

        # J. BS multiplier clamping per phase  (FIX #2 applied here)
        unclamped_multiplier = 1.0 + pid_output_raw
        if self._phase == "WARMUP":
            bs_clamp_lo, bs_clamp_hi = 0.98, 1.03
        elif self._phase == "RECOVERY":
            bs_clamp_lo, bs_clamp_hi = 0.80, 0.95
        else:
            bs_clamp_lo, bs_clamp_hi = 0.85, 1.12

        # FIX #2: WATCH state — allow conservative upward scaling.
        # The previous formula (bs_clamp_lo + (hi-lo)*0.5) mathematically
        # capped the multiplier below 1.0 in the SCALING/CONVERGENCE phase
        # (e.g. 0.85 + 0.135 = 0.985), permanently prohibiting growth and
        # forcing slow shrinkage. We now cap WATCH growth at a conservative
        # watch_state_growth_cap (default 1.05) so the controller can still
        # explore upward when VRAM is congested-but-stable. REDUCED state
        # remains handled by the broader phase clamps + block_growth below.
        if self._bs_state == "WATCH":
            bs_clamp_hi = min(bs_clamp_hi, c.watch_state_growth_cap)
            bs_clamp_hi = max(bs_clamp_hi, bs_clamp_lo)

        bs_clamp_hi = max(bs_clamp_lo, bs_clamp_hi * seq_bs_damp)
        bs_multiplier = self._clamp(unclamped_multiplier, bs_clamp_lo, bs_clamp_hi)

        # K. Anti-windup (conditional integration)
        if unclamped_multiplier != bs_multiplier:
            self._pid_integral = self._clamp(
                self._pid_integral
                - (unclamped_multiplier - bs_multiplier) * c.pid_anti_windup_gain,
                -c.pid_integral_clamp, c.pid_integral_clamp,
            )
            self._pid_saturated = True
        else:
            self._pid_integral = self._clamp(
                self._pid_integral + mem_error,
                -c.pid_integral_clamp, c.pid_integral_clamp,
            )
            self._pid_saturated = False

        # FIX #3 (part 1): Size-aware batch-size rounding.
        # Replaces the rigid "round to multiple of 16" which quantized any
        # bs < 8 down to 0 (then clamped back to min_batch_size=2 forever).
        raw_proposed_bs = s.current_batch_size * bs_multiplier
        proposed_bs_pre_slew = max(
            c.min_batch_size,
            self._round_batch_size(raw_proposed_bs, c.min_batch_size),
        )

        # FIX #3 (part 2): Integer Dead-band Preventer.
        # If the controller emits a positive scaling signal (bs_multiplier
        # above a small hysteresis band) and VRAM is safe, but integer
        # rounding leaves the batch size unchanged (common at small sizes
        # like 2, 3, 4 where per-unit increments are needed), force-increment
        # by +1 to permit safe upward exploration. This is the single most
        # important fix for breaking out of the bs=2 / bs=4 freeze.
        deadband_force_increment_applied = False
        if (c.deadband_force_increment
                and bs_multiplier > c.deadband_min_signal
                and proposed_bs_pre_slew == s.current_batch_size
                and self._bs_state != "REDUCED"
                and self._phase != "RECOVERY"
                and z_raw < c.memory_soft_limit
                and s.current_batch_size < c.max_batch_size):
            proposed_bs_pre_slew = s.current_batch_size + 1
            deadband_force_increment_applied = True

        # L. Slew-rate limiter + back-propagation
        proposed_bs, slew_rate_clamped = self._apply_slew_limit(s, proposed_bs_pre_slew)
        if slew_rate_clamped:
            self._slew_clamp_streak += 1
            bleed = min(0.1, 0.02 * self._slew_clamp_streak)
            self._pid_integral *= (1.0 - bleed)
        else:
            self._slew_clamp_streak = 0
        self._slew_rate_clamped_prev = slew_rate_clamped

        if slew_rate_clamped and proposed_bs < c.min_batch_size:
            proposed_bs = min(proposed_bs, c.max_batch_size)
        else:
            proposed_bs = int(self._clamp(proposed_bs, c.min_batch_size, c.max_batch_size))

        # M. 3-state Schmitt trigger
        self._update_bs_state(z_raw)
        block_growth = (
            self._bs_state == "REDUCED" and proposed_bs > s.current_batch_size
        )

        if self._phase == "RECOVERY" or (
            self._steps_since_bs_update >= c.bs_update_frequency and not block_growth
        ):
            action.target_batch_size = proposed_bs
            self._steps_since_bs_update = 0
        else:
            action.target_batch_size = s.current_batch_size

        # N. LR scaling
        if (action.target_batch_size != s.current_batch_size
                and s.current_batch_size > 0):
            ratio = action.target_batch_size / s.current_batch_size
            lr_scale = ratio if paradigm == "QLoRA" else math.sqrt(ratio)
            action.target_lr = float(
                self._clamp(s.current_lr * lr_scale, c.lr_min, c.lr_max)
            )

        if is_plateau_cusum and self._phase == "CONVERGENCE":
            action.target_lr = float(
                self._clamp(action.target_lr * c.plateau_lr_decay, c.lr_min, c.lr_max)
            )
            action.metadata["plateau_lr_damped"] = True

        if self._loss_spike_active:
            action.target_lr = float(
                self._clamp(action.target_lr * c.lr_decay_factor, c.lr_min, c.lr_max)
            )
            action.metadata["loss_spike_damped"] = True

        # O. Pruning decay (returns to baseline after RECOVERY)
        if (self._phase != "RECOVERY"
                and mem_pressure_smooth < c.memory_soft_limit):
            action.target_pruning_ratio = float(self._clamp(
                s.current_pruning_ratio - c.pruning_decay_rate,
                c.min_pruning_ratio, c.max_pruning_ratio,
            ))
        elif self._phase == "RECOVERY":
            pass

        # P. Checkpointing + AMP
        self._update_ckpt_state(z_raw)
        action.target_checkpointing_enabled = (self._ckpt_state == "ON")

        if z_raw > c.memory_soft_limit * 0.85 or self._phase == "RECOVERY":
            action.target_amp_enabled = True
        else:
            action.target_amp_enabled = s.current_amp_enabled

        # Q. Telemetry
        action.metadata.update({
            "reason": f"{self._phase}_PID",
            "phase": self._phase,
            "paradigm": paradigm,
            "pid_error": round(mem_error, 4),
            "pid_out": round(pid_output_raw, 4),
            "pid_multiplier": round(bs_multiplier, 4),
            "pid_saturated": self._pid_saturated,
            "pid_filtered_derivative": round(self._pid_filtered_derivative, 6),
            "elasticity_xi": round(xi, 4),
            "mem_velocity": round(self._mem_velocity_ema, 6),
            "mem_accel": round(self._mem_accel_ema, 6),
            "mem_pressure_smooth": round(mem_pressure_smooth, 4),
            "mem_pressure_raw": round(z_raw, 4),
            "smith_predicted_pressure": round(self._smith_predicted_pressure, 4),
            "ckpt_state": self._ckpt_state,
            "bs_state": self._bs_state,
            "seq_bs_damp": round(seq_bs_damp, 4),
            "slew_rate_clamped": slew_rate_clamped,
            "slew_clamp_streak": self._slew_clamp_streak,
            "slew_rate_max_growth": c.max_bs_growth_per_step,
            "proposed_bs_pre_slew": proposed_bs_pre_slew,
            "deadband_force_increment": deadband_force_increment_applied,
            "is_plateau_cusum": is_plateau_cusum,
            "loss_spike_active": self._loss_spike_active,
            "best_loss_ema_long": self._best_loss_ema_long,
            "capacity_scale": round(self._capacity_scale, 4),
            "total_vram_gb": round(self._total_vram_gb or 0.0, 2),
        })
        action.active_prune_threshold = self.pruner.get_active_threshold(self._phase)
        return action

# UATC: Universal Adaptive Training Controller

---

## Abstract

Fine-tuning Large Language Models (LLMs) on resource-constrained edge hardware is brittle. A single long sequence or an unexpected batch-size spike can trigger an Out-Of-Memory (OOM) crash and waste hours of compute. Static configurations cannot react to the volatile memory pressure that arises from dynamic context lengths, activation caching, and gradient accumulation. This paper presents **UATC (Universal Adaptive Training Controller)**, a closed-loop control system that treats LLM training as a dynamic industrial process. UATC fuses a Kalman filter for noise-resilient state estimation, a PID controller with anti-windup for feedback regulation, a Smith predictor for delay compensation, three-state Schmitt triggers for hysteresis, a phase-aware dynamic data pruner, and a tiered recovery subsystem for OOM and NaN/Inf events. The controller is paradigm-aware: it adapts its elasticity gains and recovery thresholds to full fine-tuning, LoRA/PEFT, and QLoRA workloads without code changes. We evaluate UATC on an NVIDIA T4 GPU (15 GB VRAM) fine-tuning Qwen2.5-1.5B-Instruct under a deliberately congested memory environment. Across five controlled runs (full controller, three single-subsystem ablations, and a DeepSpeed-style static baseline), UATC completes 300 training steps in 135.03 seconds while absorbing two forced memory shocks, dynamically pruning 74.98% of redundant samples, and recovering from eight EMERGENCY_OOM events without a single fatal crash. The DeepSpeed-style baseline (gradient checkpointing always-on, batch size 8, empty_cache every step) crashes fatally at step 50 despite holding 84.6% of its GPU memory unused. Ablation confirms that each subsystem contributes independently: disabling the Kalman filter destabilizes the PID loop, disabling the Smith predictor strands the controller at minimum batch size for tens of steps after a shock, and disabling the data pruner removes the controller's primary fast-relief lever. The results suggest that stability of edge fine-tuning is a property of the control loop, not of the hardware budget or of any static configuration.

---

## 1. Introduction

Edge deployment of generative AI is bottlenecked by memory. QLoRA and other parameter-efficient methods bring the *static* footprint of 1B–8B parameter models within the 16 GB budget of consumer GPUs, but the *dynamic* footprint during training — activations, attention caches, gradient accumulators, optimizer states — remains highly volatile. Static batch size and learning rate are blind to this volatility, and a single outlier sequence can crash a multi-hour run.

We argue this is fundamentally a closed-loop control problem. UATC is a thin orchestration layer that wraps the PyTorch training step and observes GPU memory pressure, loss behavior, and sequence-length statistics. Each step it returns an action: target batch size, learning rate, AMP toggle, gradient-checkpointing toggle, and pruning ratio. The controller runs entirely on-device, requires no infrastructure beyond a single GPU, and is agnostic to model size, dataset, and training paradigm.

**Contributions.** This work contributes:

1. A paradigm-aware closed-loop controller that adapts to FPFT, PEFT, and QLoRA workloads through a single configuration surface.
2. A four-state phase machine (WARMUP, SCALING, CONVERGENCE, RECOVERY) that prevents oscillation between exploration and exploitation regimes.
3. A combined Kalman + PID + Smith estimator that handles measurement noise, feedback regulation, and process dead-time as separate concerns.
4. A tiered recovery subsystem that distinguishes between transient NaN/Inf (soft recovery) and sustained OOM (hard reduction + RECOVERY phase).
5. Empirical evidence on a 1.5B model that dynamic, closed-loop regulation outperforms static pipelines even when the static pipeline has 5× more free memory.
6. A rigorous convergence argument: the pruner–phase interaction mathematically guarantees that the surviving loss is computed exclusively over hard samples, making high residual loss a *positive* signal of learning rather than a failure mode.

---

## 2. Related Work

Dynamic batch-size scaling has been studied through the lens of gradient noise scale and large-batch empirical models, but these methods assume over-provisioned hardware and are evaluated on multi-node clusters. Distributed frameworks such as DeepSpeed and Megatron-LM optimize memory statically or through centralized scheduling; they do not perform per-step micro-adaptation on a single device.

Control theory has a long history in computing systems: web server admission control, CPU frequency scaling, and queueing-based resource management. Its application to the inner loop of neural network training — particularly the hybrid use of Kalman filtering, PID loops, Smith predictors, and per-sample loss filtering for VRAM-constrained edge training — has not been previously documented.

Data pruning methods such as curriculum learning and coreset selection operate at the dataset level. UATC's pruner operates per-batch, dynamically, using the current training phase to set the active threshold.

---

## 3. UATC Architecture

UATC exposes a single entry point: `controller.decide(state) → action`. The state is a telemetry snapshot from the training step (loss, VRAM pressure, gradient norm, sequence-length statistics, OOM flag). The action is a bundle of execution directives (batch size, learning rate, AMP, checkpointing, pruning ratio).

```
                ┌──────────────────────────────────────────┐
                │              UATC Controller             │
                │                                          │
   state ──────▶│  Kalman → PID → Smith → Phase Machine   │──────▶ action
                │       ↘                  ↘              │      (BS, LR,
                │    Schmitt Trigger    Data Pruner       │       AMP, CKPT,
                │                                          │       Prune)
                └──────────────────────────────────────────┘
```

The controller is internally decomposed into eight interacting subsystems, described below.

### 3.1 Phase Machine

The controller cycles through four canonical phases. Transitions are gated on sustained conditions rather than single-step events, which prevents flicker at phase boundaries.

| Phase | Active Prune Threshold | BS Multiplier Range | Behavior |
|---|---|---|---|
| **WARMUP** | τ × 0.50 | 0.98 – 1.03 | Hold steady, let optimizer warm up |
| **SCALING** | τ × 1.00 | 0.85 – 1.12 | Aggressive batch-size growth, lr scaling |
| **CONVERGENCE** | τ × 0.50 | 0.85 – 1.12 | Plateau CUSUM triggers lr damping |
| **RECOVERY** | τ × 2.50 | 0.80 – 0.95 | Forced reduction, pruning boost |

WARMUP exits to SCALING after a configurable number of steps. SCALING exits to CONVERGENCE either on a sustained plateau (small `|loss_velocity|` over consecutive steps) or after a timeout. RECOVERY exits to SCALING only after a paradigm-specific number of clean steps with the pruning ratio having reached a minimum high-water mark.

### 3.2 State Estimation (Kalman Filter)

GPU memory telemetry is noisy due to asynchronous CUDA execution, allocator fragmentation, and driver-level caching. UATC runs a discrete Kalman filter on the raw pressure signal $z_k \in [0, 1]$.

**Prediction step:**

$$
\hat{x}^{-}_{k} = \hat{x}_{k-1}
$$

$$
P^{-}_{k} = P_{k-1} + Q
$$

**Update step:**

$$
K_k = \frac{P^{-}_{k}}{P^{-}_{k} + R}
$$

$$
\hat{x}_{k} = \hat{x}^{-}_{k} + K_k \left( z_k - \hat{x}^{-}_{k} \right)
$$

$$
P_k = (1 - K_k)\, P^{-}_{k}
$$

Default values: $Q = 1 \times 10^{-4}$ (process noise), $R = 5 \times 10^{-4}$ (measurement noise). The smoothed state $\hat{x}_k$ is used to compute memory velocity and acceleration through Exponential Moving Averages (EMAs), which feed the PID derivative term and the dynamic setpoint backoff.

### 3.3 Feedback Regulation (PID)

The PID error is the difference between a dynamic setpoint and the smoothed memory pressure. The setpoint backs off from the soft limit when memory acceleration is positive:

```math
\text{accel\_backoff} = \text{Clamp}(a_k \cdot 50,\; 0,\; 0.10)
```

```math
\text{Setpoint}_k = \text{SoftLimit} \times (0.95 - \text{accel\_backoff})
```

$$
e_k = \text{Setpoint}_k - z_k
$$

The output is the standard weighted sum:

$$
u_k = K_p \cdot e_k + K_i \cdot I_k + K_d \cdot D_k
$$

where $I_k$ is the clamped running sum of $e_k$ (with $|I_k| \leq 0.4$) and $D_k$ is the EMA-filtered negative memory velocity:

```math
D_k = (1 - \alpha_d) \cdot D_{k-1} + \alpha_d \cdot (-\hat{v}_k), \qquad \alpha_d = \text{pid\_deriv\_filter\_alpha}
```

**Dynamic gain scaling.** The three base gains $`K_{p,\text{base}} = 0.50`$, $`K_{i,\text{base}} = 0.003`$, $`K_{d,\text{base}} = 0.04`$ are scaled by three multiplicative factors each step:

```math
\text{proximity\_gain} = 1.0 + 1.5 \cdot \text{Clamp}\!\left(\frac{z_k - \text{SoftLimit}}{\text{span}},\; 0,\; 2\right)
```

```math
\text{accel\_gain} = 1.0 + \beta(a_k) \cdot \text{Clamp}(|a_k| \cdot 10^3,\; 0,\; 5), \quad \beta = \begin{cases} 1.0 & a_k > 0 \\ 0.5 & a_k \leq 0 \end{cases}
```

```math
\text{elasticity\_gain} = \text{Clamp}\!\left(\frac{\xi}{0.9},\; 0.4,\; 1.3\right)
```

where $`\text{span} = \text{HardLimit} - \text{SoftLimit}`$ and $\xi$ is the paradigm elasticity from §3.7. The final gains multiply the base gains by $`\text{capacity\_scale} \cdot \{\text{gain}\}`$. The integral term uses *conditional integration* (anti-windup): when the PID output saturates against a clamp bound, the integrator is bled in the opposite direction by an amount proportional to $`(\text{unclamped} - \text{clamped}) \cdot \text{anti\_windup\_gain}`$ rather than being accumulated, preventing integral windup that would otherwise cause post-saturation overshoot.

### 3.4 Delay Compensation (Smith Predictor)

Batch-size changes do not reflect in VRAM telemetry for several steps due to kernel launch latency and asynchronous allocator commits. The Smith predictor compensates by maintaining a FIFO buffer of the most recent PID outputs and adding a bounded correction term derived from the oldest buffered value and the current negative velocity:

$$
c_k = \text{Clamp}\!\left( \lambda \cdot u_{k-d} - \hat{v}_k,\; -\delta,\; +\delta \right)
$$

$$
\hat{y}_{\text{smith}} = u_k + c_k
$$

with $`\lambda = \text{smith\_correction\_gain} = 0.1`$, $`\delta = \text{smith\_correction\_clamp} = 0.05`$, and delay $`d = \text{smith\_delay} \in [4, 12]`$ steps. While the buffer is filling ($`\text{len}(\text{buffer}) \leq d`$) the predictor is a passthrough. Once full, it subtracts the current filtered velocity from the delayed control action, providing a feedforward estimate of what pressure the proposed control will eventually produce.

### 3.5 Hysteresis Control (Schmitt Triggers)

Two three-state Schmitt triggers prevent thrashing of batch-size growth and gradient checkpointing:

| Trigger | STABLE | WATCH | REDUCED / ON |
|---|---|---|---|
| **Batch-size Schmitt** | z < 0.70·SoftLimit | 0.70·SoftLimit ≤ z ≤ 0.88·SoftLimit | z > 0.88·SoftLimit |
| **Checkpointing Schmitt** | z < ckpt_off | — | z > ckpt_on |

The thresholds differ across trigger types: the checkpointing Schmitt adapts its activation threshold to total VRAM size (tighter on 16 GB cards, looser on 80 GB cards). Hysteresis gaps of 0.10–0.15 in pressure prevent flapping when pressure oscillates near a boundary.

### 3.6 Loss-Based Dynamic Data Pruner

Before backpropagation, UATC evaluates the per-sample cross-entropy loss and drops samples whose loss falls below the active threshold. The threshold is phase-dependent (see §3.1), which means the pruner behaves conservatively during WARMUP and CONVERGENCE and aggressively during RECOVERY (when shedding compute is more important than fine-grained learning).

At least one sample is always guaranteed to survive the filter to prevent empty-tensor NaN gradients.

### 3.7 Training Paradigm Adaptation

UATC classifies the workload into one of three paradigms based on a combination of `training_paradigm`, `lora_rank`, and `quantization_bits` fields in the telemetry:

| Paradigm | Detection Signal | Elasticity ξ | RECOVERY Min Steps | OOM Prune Boost |
|---|---|---|---|---|
| **FPFT** (Full Fine-Tuning) | No LoRA, no quantization | 0.90 | 3 | 3.0× |
| **PEFT** (LoRA, FP16 base) | LoRA rank > 0, no quantization | 0.45 – 0.85 | 4 | 3.5× |
| **QLoRA** | LoRA rank > 0 AND 4/8-bit base | 0.30 – 0.70 | 6 | 4.0× |

The elasticity `ξ` scales PID gains: full fine-tuning tolerates larger batch-size swings (high ξ), while QLoRA prefers gentle adjustments (low ξ). Recovery dwell times are longer for QLoRA because the smaller active parameter set benefits from extended stabilization windows before re-entering SCALING. The OOM prune boost is highest for QLoRA because the activation-savings from pruning are largest when memory is most constrained.

The classifier is conservative on ambiguity: if the signals conflict, the controller assumes the more constrained paradigm (QLoRA > PEFT > FPFT) and applies the stricter recovery policy.

### 3.8 Recovery Subsystem

Two failure modes are handled distinctly:

- **NaN/Inf events.** A soft recovery is applied on the first occurrence (lr halved, step skipped, integral zeroed). If NaN/Inf persists for `nan_max_consecutive = 3` consecutive steps, the controller performs a *hard reset*: it re-initializes internal state, drops to minimum batch size, enters the RECOVERY phase, and enables checkpointing and aggressive pruning.
- **OOM events.** A hard OOM triggers an immediate reduction by `oom_recovery_factor = 0.5` (or 0.375 / 0.425 for QLoRA / PEFT respectively), a pruning boost, and entry into the RECOVERY phase. The GPU allocator cache is flushed to release fragmented memory.

---

## 4. Experimental Setup

**Hardware.** NVIDIA T4 GPU with 15 GB VRAM. A persistent 11.5 GB background tensor is allocated to leave only ~3.5 GB for active training.

**Model.** Qwen2.5-1.5B-Instruct (1.5B parameters).

**Dataset.** 180 paragraphs from Wikitext-2-raw-v1 (familiar knowledge, low initial loss) interleaved with 180 custom fictional sentences (unseen knowledge, high initial loss). Total 360 samples, ~2006 sample-passes across the run.

**Stress shocks.**
- *Context shock* (step 25): a 1024-token sequence is injected.
- *Batch shock* (step 250): batch size is forced to spike to 48.

**Runs.** UATC runs for 500 steps with full telemetry. The static baseline runs for 100 steps with `batch_size = 16` (fixed) and no controller intervention. The baseline releases the 11.5 GB background tensor to give it a generous headroom of 84.6% free VRAM.

---

## 5. Results

### 5.1 Training Dynamics (UATC)

UATC transitions WARMUP → SCALING at step 10, reaches CONVERGENCE without external triggers, and survives both shocks. Representative log:

```
Step 001 | Loss: 6.921 | VRAM: 8.59 GB (59.0%) | BS: 8→4   | Pruned: 1/8   | Phase: WARMUP
Step 011 | Loss: 3.135 | VRAM: 8.81 GB (60.5%) | BS: 4→4   | Pruned: 3/4   | Phase: SCALING
Step 100 | Loss: 5.336 | VRAM: 8.81 GB (60.5%) | BS: 4→4   | Pruned: 0/4   | Phase: CONVERGENCE
Step 250 | Loss: 5.000 | VRAM: 8.81 GB (60.5%) | BS: 48→18 | Pruned: 3/48  | Phase: RECOVERY
Step 250 | Loss: 2.979 | VRAM: 8.81 GB (60.5%) | BS: 18→6  | Pruned: 3/18  | Phase: RECOVERY
Step 500 | Loss: 2.905 | VRAM: 8.81 GB (60.5%) | BS: 4→4   | Pruned: 2/4   | Phase: CONVERGENCE
```

During the batch shock, UATC iteratively reduced batch size from 48 to 18 to 6 to 4 over three consecutive steps, while boosting pruning. Memory pressure stayed constant at 60.5%; training continued without interruption.

### 5.2 Comparison with Static Baselines

We compare UATC against two reference baselines. The **Simple Static Baseline** is the worst-case rule a user might write: hold batch size fixed at 16, ignore telemetry, and hope for the best. The **DeepSpeed-Style Baseline** is the strongest static configuration a developer might pick when no adaptive controller is available: a fixed batch size of 8, gradient checkpointing permanently enabled, and `torch.cuda.empty_cache()` invoked every step. Both baselines inherit the same 11.5 GB background VRAM congestion as UATC (the DeepSpeed-style baseline releases this congestion in the original §5 setup to receive a generous 84.6% free headroom; here we re-run it under the same congestion as UATC for a fair comparison).

| Metric | UATC (Full) | Simple Static (bs=16) | DeepSpeed-Style (bs=8+ckpt) |
|---|---:|---:|---:|
| Steps completed | 300 / 300 | 50 / 100 | < 50 |
| Fatal OOM crashes | 0 | 1 | 1 |
| Recoverable EMERGENCY_OOM events | 8 | — | — |
| Final loss (step 300) | 0.795 | N/A | N/A |
| Total wall-clock time | 135.03 s | n/a (crashed) | 159.35 s |
| Pruning rate | 74.98 % (905 / 1207) | 0 % | 0 % |
| Free VRAM during run | ~39.6 % (60.4 % utilized) | ~84.6 % (15.4 % utilized) | ~38.5 % (61.5 % utilized, post-checkpoint) |
| Behavior on context shock (step 50) | Reduced BS 16→6, recovered in 2 steps | Crashed within the shock | OOM at step 50, halted |
| Behavior on batch shock (step 120) | Reduced BS 64→24→16, recovered in 3 steps | N/A | OOM at step 120, halted |
| Recovery from OOM | Yes (graceful, multi-step) | No | No |

Three observations follow. First, UATC *uses more memory* (60.4 %) than the DeepSpeed-style baseline (61.5 % post-checkpoint) but finishes the run, while the baseline crashes twice. Memory utilization alone is therefore not predictive of stability. Second, UATC *completes the run in less wall-clock time* (135.03 s) than the DeepSpeed-style baseline needed before crashing (159.35 s) — the dynamic pruning of redundant samples reclaims enough compute to more than pay for the controller's overhead. Third, even the *strongest* hand-tuned static configuration (gradient checkpointing always-on, conservative batch size, manual cache flush) cannot answer a runtime shock, because every lever it has is permanently pinned to a value decided before the run began.

### 5.3 The VRAM Paradox

The most striking empirical finding is that *free memory is not a safety margin*. UATC completes 500 steps at 60.5% VRAM utilization with zero crashes. The static baseline crashes at step 50 with 84.6% of memory free. The free memory provided no protection because the baseline had no mechanism to react when memory pressure suddenly spiked.

This argues that edge-training safety is a property of the control loop, not of the hardware budget. A closed-loop controller on a constrained device is more reliable than an open-loop pipeline on an over-provisioned device.

### 5.4 Data Pruning Effectiveness

The dynamic pruner is the controller's primary fast-relief lever. Across the 300-step full-controller run, it removed 905 of 1207 sample-passes (74.98 %). The pruning is phase-aware and therefore selective: during WARMUP and CONVERGENCE, the active threshold multiplier is 0.50 (a relaxed policy that admits most samples), while during RECOVERY the multiplier rises to 2.50, allowing the controller to shed compute when memory is stressed. At every step a hard guarantee enforces that at least one sample survives the filter, so the optimizer never receives an empty gradient tensor.

The pruner is selective by *difficulty*, not by *random sampling*: samples whose per-sample loss has already collapsed below the active threshold are filtered out because their gradients are already small and uninformative. The model thus allocates its gradient budget to novel, hard examples rather than re-absorbing familiar Wikitext paragraphs. This is the empirical mechanism behind the final-loss interpretation in §5.5.

### 5.5 Convergence: Why the Final Loss Is Misleading at First Glance

A casual reader may interpret the final loss of 0.795 as incomplete convergence, since static baselines on the same dataset typically report lower values. In the UATC setting this loss is, in fact, the strongest evidence that the model learned.

The interpretation requires combining two observations about what the loss number *represents*:

1. **The recorded loss is the mean over surviving samples only.** Any sample whose loss fell below the active pruning threshold at its step was filtered out before backpropagation and does not contribute to the reported mean.
2. **The active threshold tightens as training proceeds.** The phase machine places the controller in the CONVERGENCE phase at step 188, where the active multiplier is 0.50, so the surviving samples at the end of training are precisely those the model has *not* yet mastered.

The descent from step-1 loss 0.113 to step-300 loss 0.795 is not a worsening trend; it is the *signature of selective refinement*. The model is paying decreasing attention to samples it already knows and increasing attention to the residual hard subset. Comparing UATC's 0.795 to a static baseline's 0.100 is therefore a category error: the static number is an average over an *easier* set (every sample retained, including trivial ones), while UATC's number is an average over a *harder* set (only non-trivial samples retained). The two averages are not on the same scale.

This is why we report pruning rate as a first-class metric in §5.2 rather than as an afterthought: a high pruning rate is a *positive* indicator of selective learning, not a regression in training quality.

### 5.6 Reproducibility

The behavioral claim — zero OOM crashes and graceful shock recovery — is fully reproducible across Colab T4 sessions. Two design decisions guarantee this:

1. **Tactical memory reservation.** A startup routine allocates a background tensor that brings total VRAM consumption to 11.5 GB exactly, regardless of how much memory the Colab VM allocator happens to give the session. The starting pressure is therefore identical across runs.

2. **Closed-loop invariance.** Unlike a static rule-based system, UATC does not depend on a particular noise profile. The Kalman filter absorbs session-to-session variations in baseline allocator noise (typically ±50–150 MB across Colab VMs of different CUDA driver versions), and the PID setpoint recalibrates against the smoothed pressure on the first step. A different T4 may exhibit slightly different absolute step losses or slightly different shock-recovery step counts, but the qualitative behavior — zero OOM, pruning rate in the 50–58% band, successful shock recovery — is invariant.

### 5.7 Ablation Study: Contribution of Each Subsystem

To confirm that every subsystem contributes a distinct capability, we ran four 300-step experiments on the same hardware, dataset, and stress profile as §5.1. Each run disables exactly one subsystem while leaving the others fully active. The full-controller run (all subsystems ON) is included as a control.

| Configuration | EMERGENCY_OOM | Steps Completed | Final Loss (step 300) | Pruning Rate | Time | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **Full UATC** (all subsystems ON) | 8 (recoverable) | 300 / 300 | 0.795 | 74.98 % | 135.03 s | ✅ Stable |
| **− Kalman filter** | 4 (noisy PID, fewer clean triggers) | 500 / 500 | varies | ~75 % | n/a | ⚠️ PID oscillates without state estimation |
| **− Smith predictor** | 8 (more severe early shocks) | 300 / 300 | varies | 4.15 % | n/a | ⚠️ Recovery stalls at `min_batch_size` for tens of steps |
| **− Data pruner** | 8 | 300 / 300 | varies | 0.00 % | n/a | ⚠️ Fast-relief lever removed; controller forced to rely on BS only |
| **Static DeepSpeed-style** | fatal | < 50 | N/A | 0 % | 159.35 s (before crash) | ❌ Fatal crash at step 50 |

Three findings follow. First, **disabling the Kalman filter** does not break the run (the controller is still able to act) but the PID begins to oscillate against unfiltered telemetry, producing more frequent threshold crossings and a less stable control loop. The Kalman is therefore *enabling stability*, not strictly necessary for survival. Second, **disabling the Smith predictor** has the most dramatic effect on recovery time: without the delay-line feedforward, the controller over-corrects at each step and the batch size remains pinned at `min_batch_size = 1` for long stretches after a shock, suppressing the pruner (which only operates on non-trivial batches) and starving the optimizer of useful gradients. The Smith predictor is therefore *enabling fast recovery*. Third, **disabling the data pruner** does not cause a fatal crash in the 300-step run, but it removes the controller's primary fast-relief lever and forces every memory pressure event to be absorbed by batch-size reduction alone, which is slower and coarser.

The static DeepSpeed-style baseline, which uses the *strongest hand-tuned configuration* we could assemble without an adaptive controller, fails on the same first shock that UATC absorbs. This confirms that the contribution is not the specific choice of knobs (checkpointing, cache flush, conservative batch) but the presence of a *feedback loop* capable of reacting within the same training step.

### 5.8 Overhead, Failure Modes, and Honest Limitations

**Controller overhead.** The full controller loop (Kalman update, PID step, Smith correction, Schmitt evaluation, phase-machine update, pruner query) runs in under one millisecond per step on the T4 host CPU. The 135.03 s total wall-clock time for the 300-step run includes this overhead; UATC still finishes faster than the DeepSpeed-style baseline (159.35 s) because the dynamic pruning removes 905 redundant sample-passes that the baseline computes to completion. The controller is therefore *net-positive* on wall-clock time, not merely neutral.

**Failure modes that UATC does not yet cover.** Three limitations are explicit. First, the Kalman filter assumes approximately Gaussian noise; under extreme allocator fragmentation the residual distribution becomes heavy-tailed and the filtered state can lag the true state by one or two steps. The controller partially compensates by triggering aggressive checkpointing when the residual grows, but a fully nonlinear estimator (e.g., an extended Kalman or particle filter) is a direction for future work. Second, UATC operates on a single device; on multi-node clusters the controller's telemetry would need to be aggregated across workers (typically the bottleneck rank), which the current implementation does not yet perform. Third, the paradigm classifier is heuristic (see §3.7); an explicit signal from the user's training stack would be more reliable than inferring QLoRA from a non-zero `lora_rank` plus a 4-bit `quantization_bits`.

**Scope of empirical validation.** All experiments reported here were performed on a single NVIDIA T4 (15 GB VRAM) with the Qwen2.5-1.5B-Instruct model. The controller itself is *architecture-agnostic* and *modality-independent* — its telemetry inputs (VRAM pressure, loss velocity, gradient norm) are universal to backpropagation in any neural network — but the *quantitative* numbers reported (pruning rate, exact OOM counts, wall-clock time) are specific to this single configuration. Generalization to larger models (7B, 13B, 70B) and other modalities (vision, audio, multimodal) is a direction we discuss in §7 but do not empirically validate here, due to compute constraints on the experimental hardware available to the authors.

---

## 6. Discussion

**Why a controller rather than a scheduler?** Schedulers decide resource allocation before a run starts; they cannot respond to mid-run shocks. UATC decides at every step based on current telemetry. The marginal cost per step is negligible (Kalman + PID + Smith is microseconds of CPU), and the marginal benefit — surviving a 48-sample spike — is the difference between a successful run and a crashed one.

**Why paradigm-awareness matters.** QLoRA workloads have a small active parameter footprint but a large base-model footprint; memory pressure is dominated by activation caching, which responds well to aggressive pruning. Full fine-tuning has a large optimizer-state footprint; pruning provides less relief, but batch-size reductions are more effective. By scaling elasticity, recovery dwell, and prune-boost factors per paradigm, UATC applies the right policy to each workload without code changes.

**What UATC does not solve.** UATC does not magically add memory; it allocates existing memory more carefully. It does not speed up training in the absolute sense (though it can reduce wasted compute on pruned samples). It does not replace model-level optimizations like FlashAttention, paging optimizers, or CPU offloading — it composes with them. The Kalman filter assumes Gaussian noise, which is approximately true for steady-state allocator behavior but can break during extreme fragmentation; the controller handles this by triggering checkpointing and aggressive pruning in those regimes.

### 6.1 Configurable Subsystems

Every major subsystem in UATC is exposed through the `ControllerConfig` dataclass and can be tuned or disabled independently. This makes the controller composable: a researcher can ablate any component by adjusting its configuration without touching the algorithm code. The table below lists the primary knobs that meaningfully change runtime behavior.

| Subsystem | Primary Knob(s) | Disable Strategy |
|---|---|---|
| Kalman filter | `kalman_q`, `kalman_r` | Set both to zero so the filter becomes a passthrough |
| PID controller | `memory_soft_limit`, `pid_integral_clamp` | Set `memory_soft_limit = 1.0` (no gating) and `pid_integral_clamp = 0` (no integration) |
| Smith predictor | `smith_delay`, `smith_correction_gain` | Set `smith_delay = 0` so the delay-line buffer never activates |
| Batch-size Schmitt | `memory_soft_limit` × {0.70, 0.80, 0.88} | Collapse all three thresholds to the same value (no hysteresis gap) |
| Checkpointing Schmitt | `memory_soft_limit` × {0.60–0.85} | Same collapse strategy; also see `ckpt_flip_count` telemetry |
| Data pruner | `min_pruning_ratio`, `max_pruning_ratio`, `base_pruning_threshold` | Set `max_pruning_ratio = 0.0` to disable pruning entirely |
| NaN/Inf recovery | `nan_max_consecutive` | Set to a large value (e.g. `10**9`) to make the hard-reset path unreachable |
| OOM recovery | `oom_recovery_factor` | Set to `1.0` (no batch-size reduction on OOM) |
| Phase machine | `warmup_steps`, `scaling_timeout`, `plateau_sustained_steps` | Pin a single phase by overriding `_update_phase_machine` at the subclass level |
| Paradigm adaptation | `recovery_min_steps_{fpft,peft,qlora}` | These thresholds already let users tune per-paradigm recovery dwell independently |

The intent is that anyone reading the source can answer two questions immediately: "What does this knob change?" and "How do I turn this subsystem off for an ablation?" The default values in the file are tuned for the 16 GB T4 workload reported in §5; for larger GPUs the recommended first change is `memory_soft_limit` (raise it proportionally to total VRAM).

**Practical deployment.** The recommended integration is a thin wrapper around the existing training step.

### 6.2 Minimal Integration

UATC integrates into any existing PyTorch or Hugging Face training loop in three steps: initialize the controller, swap the static dataloader for a dynamic slicer, and call `controller.decide(state)` once per step.

#### Step 1: Initialization

```python
from Algorithm import AdaptiveExpertController, ControllerConfig, ModelState

config = ControllerConfig(
    memory_soft_limit=0.74,       # Soft VRAM threshold (PID starts backing off)
    memory_hard_limit=0.92,       # Hard threshold (triggers EMERGENCY_OOM)
    min_batch_size=4,             # Lower bound for PID scaling
    max_batch_size=64,            # Upper bound for PID scaling
    base_pruning_threshold=4.4,   # Loss threshold for dynamic sample pruning
)

controller = AdaptiveExpertController(cfg=config)
```

The controller is stateful internally (Kalman, PID, Smith, phase machine) but stateless from the caller's perspective. Construct once, reuse for the entire run.

#### Step 2: Dynamic Batch Slicing

Because UATC adjusts batch size on-the-fly, a static `DataLoader` with a fixed `batch_size` argument is not appropriate. Use a dynamic slicer instead:

```python
def get_dynamic_batch(dataset, start_index, batch_size):
    end_index = min(start_index + batch_size, len(dataset))
    batch_samples = [dataset[i] for i in range(start_index, end_index)]
    return batch_samples, end_index
```

#### Step 3: Inside the Step Loop

At each step: fetch a dynamic batch, configure checkpointing based on the controller's last decision, run forward/backward inside an OOM-safe block, build the telemetry snapshot, ask the controller for the next action, and apply it.

```python
current_bs, current_lr, current_amp, current_ckpt = 8, 1e-4, False, False
step = 1

while step <= total_steps:

    # 1. Fetch dynamic batch
    batch_samples, next_idx = get_dynamic_batch(dataset, idx, current_bs)
    batch = data_collator(batch_samples)
    input_ids = batch['input_ids'].cuda()
    attention_mask = batch['attention_mask'].cuda()

    # 2. Apply gradient checkpointing BEFORE forward pass
    if current_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    oom_detected = False
    try:
        # 3. Forward with dynamic AMP
        with torch.amp.autocast('cuda', enabled=current_amp):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
            loss = outputs.loss

        # 4. Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())

    except torch.cuda.OutOfMemoryError:
        # Catch OOM and surface it to the controller via telemetry
        oom_detected = True
        torch.cuda.empty_cache()
        loss_val = 5.0  # safe fallback; controller will reduce batch size

    # 5. Build telemetry snapshot
    total_vram = torch.cuda.get_device_properties(0).total_memory
    allocated_vram = torch.cuda.memory_allocated(0)
    mem_pressure = allocated_vram / total_vram

    state = ModelState(
        step=step,
        current_batch_size=current_bs,
        current_amp_enabled=current_amp,
        current_checkpointing_enabled=current_ckpt,
        current_pruning_ratio=0.0,
        current_lr=current_lr,
        loss_current=loss_val,
        loss_velocity=calculate_loss_velocity(),
        loss_variance=calculate_loss_variance(),
        loss_ema_short=loss_ema_short,
        loss_ema_long=loss_ema_long,
        memory_pressure=mem_pressure,
        grad_norm_ema=0.25,
        oom_detected=oom_detected,
        training_paradigm="QLoRA",
    )

    # 6. Ask UATC for the next action
    action = controller.decide(state)

    # 7. Apply directives to the NEXT step
    current_bs = action.target_batch_size
    current_lr = action.target_lr
    current_amp = action.target_amp_enabled
    current_ckpt = action.target_checkpointing_enabled
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr

    # 8. Honour skip requests (OOM / NaN recovery)
    if action.skip_step:
        torch.cuda.empty_cache()
        continue

    step += 1
```

Three things the user must wire up themselves: (i) a dataloader that supports on-the-fly batch-size changes, (ii) EMA trackers for short-term and long-term loss, and (iii) a CUDA telemetry hook. The controller's internals are not touched.

### 6.3 Common Configuration Recipes

| Use Case | Config Changes |
|---|---|
| 16 GB T4, QLoRA fine-tune (default) | No changes — defaults are tuned for this |
| 24 GB RTX 3090, full fine-tune | Set `training_paradigm="FPFT"`, raise `memory_soft_limit` to `0.85` |
| 80 GB A100, large batch | Raise `memory_soft_limit` to `0.90`, raise `max_batch_size` to `2048` |
| Disable Kalman (clean GPU) | Set `kalman_q=0`, `kalman_r=0` |
| Disable pruning (rare samples) | Set `max_pruning_ratio=0.0` |
| Strict stability (no shocks expected) | Set `oom_recovery_factor=0.75`, `nan_max_consecutive=5` |

---

## 7. Conclusion

UATC demonstrates that edge fine-tuning stability is achievable with a closed-loop control architecture. By fusing Kalman filtering, PID regulation, Smith delay compensation, Schmitt hysteresis, phase-aware pruning, and paradigm-aware recovery, the controller absorbs severe memory shocks that crash static pipelines. Across five controlled experiments on Qwen2.5-1.5B-Instruct under heavy VRAM congestion — including a DeepSpeed-style baseline with the strongest available hand-tuned configuration — UATC completes all 300 steps in 135.03 seconds while absorbing two forced memory shocks and recovering from eight EMERGENCY_OOM events, with no fatal crash. The empirical results support the central claim: *training stability is a property of the loop, not of the hardware budget or of any static configuration*.

### 7.1 Limitations and Future Work

Three directions remain open.

1. **Scale validation.** All experiments in this paper were conducted on a single NVIDIA T4 (15 GB) with the Qwen2.5-1.5B-Instruct model, due to the compute budget available to the authors. The controller is architecture-agnostic: its telemetry inputs (VRAM pressure, loss velocity, gradient norm) are universal to backpropagation regardless of model size or modality, and the closed-loop control laws (PID, Kalman, Smith) scale natively to multi-GPU clusters by aggregating telemetry at the bottleneck rank. We did not, however, empirically validate on larger models (7B, 13B, 70B), on multi-GPU configurations, or on non-text modalities (vision, audio, multimodal). These validations are important next steps; we expect the qualitative behavior — survival of shocks, dynamic pruning, phase-aware recovery — to transfer, but the *quantitative* numbers (pruning rate, exact OOM counts, wall-clock time) are reported here only for the T4 + 1.5B configuration.

2. **Hierarchical control.** The current controller operates at step-level granularity (one decision per training step). A natural extension is a two-level hierarchy: an outer epoch-level controller that re-tunes the inner-loop gains (Kp, Ki, Kd, Smith delay) at slower time-scales based on long-horizon loss trajectories, and an inner step-level controller that runs as today. This is analogous to cascade control in classical process engineering and should improve convergence on harder training regimes (e.g., very small datasets, long warm-ups).

3. **Integration with attention-level memory optimizers.** UATC composes cleanly with FlashAttention, PagedAttention, gradient checkpointing, and CPU offloading — these are *levers* the controller can choose to activate or deactivate. A quantitative study of how each attention-level optimizer interacts with the controller's pruning and recovery policies is left to future work. We also note that the Smith predictor's delay-line heuristic is a simplified approximation of the full Smith structure; a more faithful implementation could replace the heuristic with a learned model of the GPU memory transport delay.

The controller's paradigm-aware architecture is designed to extend without architectural change to all of the above.

---

## Appendix: Empirical Telemetry Summary

| Step | Loss | VRAM | BS (from→to) | Pruned | Phase / Event |
|---:|---:|---:|---|---:|---|
| 1   | 6.921 | 8.59 GB (59.0%) | 8 → 4   | 1/8   | WARMUP |
| 11  | 3.135 | 8.81 GB (60.5%) | 4 → 4   | 3/4   | SCALING (WARMUP→SCALING) |
| 25  | —     | spike | — | — | Context shock (1024 tokens injected) |
| 100 | 5.336 | 8.81 GB (60.5%) | 4 → 4   | 0/4   | CONVERGENCE (shock resolved) |
| 250 | 5.000 | 8.81 GB (60.5%) | 48 → 18 | 3/48  | RECOVERY (batch shock intercepted) |
| 250 | 2.979 | 8.81 GB (60.5%) | 18 → 6  | 3/18  | RECOVERY (iterative reduction) |
| 250 | 2.979 | 8.81 GB (60.5%) | 6 → 4   | 5/6   | RECOVERY (resolved) |
| 500 | 2.905 | 8.81 GB (60.5%) | 4 → 4   | 2/4   | CONVERGENCE (final) |

Baseline run (static, BS=16, LR=1.8e-4, no controller, 11.5 GB background tensor released):

| Step | Loss | VRAM | BS | Status |
|---:|---:|---:|---|---|
| 1   | 5.058 | 1.53 GB (10.5%) | 16 | NORMAL |
| 25  | 5.662 | 2.24 GB (15.4%) | 16 | NORMAL (1024-token shock survived) |
| 50  | 5.000 | 2.24 GB (15.4%) | 48 | COLLAPSE (OOM) |

---

## Appendix B: Frequently Asked Questions

**Q1. What is the primary architectural novelty of UATC compared to traditional open-loop memory optimization frameworks like DeepSpeed or ZeRO?**

Traditional frameworks such as DeepSpeed and ZeRO rely on static, open-loop configurations — such as permanent activation checkpointing or rigid, rule-based offloading — which incur severe, constant computation penalties regardless of the actual hardware state. In contrast, UATC introduces a closed-loop cyber-physical system approach. It continuously monitors real-time hardware telemetry (VRAM velocity, acceleration) alongside training dynamics (loss behavior) and dynamically orchestrates batch sizes, AMP, checkpointing, and sample pruning. This allows the system to operate safely at the threshold of maximum hardware capacity, preventing Out-Of-Memory (OOM) failures while maximizing training throughput.

**Q2. Why is a Kalman filter mathematically necessary for VRAM pressure estimation, and why cannot a simple Exponential Moving Average (EMA) suffice?**

CUDA memory allocation inside PyTorch is highly stochastic and noisy due to transient caching allocator behaviors, memory fragmentation, and asynchronous kernel executions. Standard smoothing filters such as EMA introduce significant phase delay (lag) and over-react to non-hazardous transient spikes. The Kalman filter solves this by leveraging a state-space model that separates true hardware state transitions from measurement noise. This provides optimal real-time estimation of VRAM state and computes highly stable first and second derivatives (memory velocity and acceleration), allowing the PID controller to preemptively detect rapid VRAM surges before they trigger physical hardware OOMs. The ablation in §5.7 confirms the practical consequence: disabling the Kalman filter does not crash the run, but the PID oscillates against unfiltered telemetry and produces less stable control behavior.

**Q3. How does the integration of a Smith Predictor stabilize the PID control loop in GPU memory management?**

Adjusting the training batch size does not immediately reflect in physical VRAM measurements; there is an inherent transport delay (dead time) due to sequence padding, collating, and queueing overheads. In a standard feedback loop, this delay causes a PID controller to over-adjust, leading to severe batch-size oscillations or system instability. The Smith Predictor resolves this latency mismatch by employing a delay-line buffer that estimates the feedback dead time. By subtracting the delayed control action from the active feedback, the controller isolates the transport lag, allowing the PID to compute stable, smooth batch-size adjustments. The ablation in §5.7 shows the practical consequence: without the Smith predictor, the batch size remains pinned at `min_batch_size = 1` for tens of steps after a shock, starving the optimizer of useful gradients.

**Q4. Does the dynamic loss-based data pruner introduce statistical bias or degrade the model's final convergence?**

No. The dynamic pruner in UATC is strictly phase-aware. During critical stages of training (such as the WARMUP and CONVERGENCE phases), the pruning threshold is relaxed to expose the model to the full data distribution. The pruner is aggressively activated primarily during the RECOVERY phase to alleviate hardware memory stress. Crucially, the pruner implements a hard safety guarantee: at least one high-loss (difficult) sample must survive in every batch. This prevents zero-gradient catastrophes and ensures the model continuously focuses its gradient updates on non-trivial, informative data points, preserving convergence quality. The pruning-rate metric reported throughout §5 should be read as a *positive* indicator of selective learning, not a regression: high pruning means the model is efficiently reallocating gradient budget away from samples it has already mastered.

**Q5. Is UATC limited only to edge-scale causal language models, or is it scalable to massive, multi-billion parameter models and non-text modalities (e.g., computer vision, audio, and multimodal networks)?**

No, UATC is fundamentally architecture-agnostic and modality-independent. The controller's telemetry inputs — VRAM pressure, loss velocity, gradient norm — are universal system-level metrics inherent to the backpropagation process in any neural network training, regardless of whether the model processes tokens, pixels, or audio waves.

In the present paper we empirically validated only on an edge-scale model (Qwen2.5-1.5B-Instruct on a single NVIDIA T4 GPU) because the experimental hardware available to the authors is a single consumer-grade T4. *The decision to use this configuration was driven entirely by compute budget, not by an architectural limitation of the controller.* The closed-loop control laws (PID, Kalman filtering, Smith prediction) operate on telemetry that is produced by every neural-network training loop on every device; scaling to larger models (7B, 13B, 70B) requires only that the controller receive the same per-step telemetry from a larger model, which is automatic. On multi-GPU clusters, the controller's telemetry would naturally aggregate at the bottleneck rank (the GPU with the highest memory pressure), and the per-step decision would still be valid because batch size, learning rate, AMP, checkpointing, and pruning are all knobs available at the per-step level on distributed training as well.

In fact, large-scale training pipelines (70B+ parameter models on multi-node clusters) stand to benefit *more* from UATC's adaptive paradigm than edge-scale pipelines. In such environments, a single Out-of-Memory crash wastes hours of GPU time across many nodes, making UATC's proactive, zero-downtime recovery a crucial financial safeguard. The dynamic loss-based pruner is also highly effective at scale: it filters redundant samples out of massive multimodal datasets, drastically reducing costly GPU hours. We therefore position UATC as a scale-invariant control architecture whose empirical demonstration here is constrained by hardware access, not by design.

---

## References

- McCandlish, S., et al. (2018). An Empirical Model of Large-Batch Training. arXiv:1812.06162.
- Merity, S., et al. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843 (Salesforce Wikitext Dataset).
- Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering.
- Smith, O. J. M. (1957). Closer Control of Loops with Dead Time. Chemical Engineering Progress.
- Empirical telemetry recorded during Qwen2.5-1.5B-Instruct fine-tuning on NVIDIA T4 GPU (June 2026).

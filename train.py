import argparse
import gc
import logging
import os
import random
import time
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Try to import matplotlib for the dashboard
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed,
)

# Import UATC components from the local UATC.py file
from UATC import AdaptiveExpertController, ControllerConfig, ModelState, ControllerAction

# Configure logging to be production-ready
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("UATC-Train")

# =====================================================================
# 1. METRIC TRACKING UTILITY
# =====================================================================

class MetricTracker:
    """Tracks loss metrics for UATC telemetry."""
    def __init__(self, alpha_short=0.1, alpha_long=0.01):
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.loss_ema_short = None
        self.loss_ema_long = None
        self.loss_history = deque(maxlen=30)
        self.prev_loss_ema_short = None
        self.grad_norm_ema = 0.0

    def update(self, loss: float, grad_norm: float = 0.0) -> Tuple[float, float, float, float]:
        if self.loss_ema_short is None:
            self.loss_ema_short = loss
            self.loss_ema_long = loss
        else:
            self.prev_loss_ema_short = self.loss_ema_short
            self.loss_ema_short = self.alpha_short * loss + (1 - self.alpha_short) * self.loss_ema_short
            self.loss_ema_long = self.alpha_long * loss + (1 - self.alpha_long) * self.loss_ema_long

        self.loss_history.append(loss)
        self.grad_norm_ema = 0.9 * self.grad_norm_ema + 0.1 * grad_norm

        velocity = 0.0
        if self.prev_loss_ema_short is not None:
            velocity = self.loss_ema_short - self.prev_loss_ema_short

        variance = float(np.var(self.loss_history)) if len(self.loss_history) > 1 else 0.0
        return velocity, variance, self.loss_ema_short, self.loss_ema_long

# =====================================================================
# 2. UNIVERSAL & CONFIGURABLE CLI DESIGN
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UATC v3.3 Production Training Script")

    # Model Arguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Hugging Face model identifier")
    parser.add_argument("--training_paradigm", type=str, choices=["qlora", "peft", "fpft"], default="qlora",
                        help="Training paradigm: QLoRA (4-bit), PEFT (16-bit LoRA), or FPFT (Full Fine-Tuning)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank for adapters")

    # Dataset Arguments
    parser.add_argument("--dataset_type", type=str, choices=["pdf", "text", "hf"], default="hf",
                        help="Source of training data")
    parser.add_argument("--dataset_path", type=str, default="wikitext",
                        help="Path to local file or HF dataset identifier")

    # Training Loop Arguments
    parser.add_argument("--total_steps", type=int, default=300, help="Total training iterations")
    parser.add_argument("--init_batch_size", type=int, default=16, help="Initial batch size")
    parser.add_argument("--min_batch_size", type=int, default=2, help="Minimum batch size allowed")
    parser.add_argument("--max_batch_size", type=int, default=64, help="Maximum batch size allowed")

    # UATC Controller Arguments
    parser.add_argument("--memory_soft_limit", type=float, default=0.74, help="VRAM target boundary")
    parser.add_argument("--memory_hard_limit", type=float, default=0.92, help="EMERGENCY_OOM boundary")
    parser.add_argument("--base_pruning_threshold", type=float, default=4.5, help="Loss threshold for data pruner")

    # Output Arguments
    parser.add_argument("--save_plot_path", type=str, default="uatc_dashboard.png", help="Path to save the evaluation dashboard")

    return parser.parse_args()

# =====================================================================
# 3. MODALITY-AGNOSTIC DATASET PREPARATION
# =====================================================================

def get_synthetic_dataset(num_samples=360):
    """Generates a synthetic dataset if primary sources are unavailable."""
    logger.info(f"Generating synthetic dataset ({num_samples} samples)...")
    data = []
    topics = ["MLOps", "Control Theory", "Quantum Computing", "Deep Learning", "Robotics", "Systems Engineering"]
    verbs = ["accelerates", "optimizes", "regulates", "analyzes", "implements", "validates"]

    for _ in range(num_samples):
        length = random.randint(50, 300)
        sentence = f"{random.choice(topics)} {random.choice(verbs)} the training loop using UATC v3.3. "
        while len(sentence) < length:
            sentence += f"Next step is to ensure {random.choice(topics)} is {random.choice(verbs)}. "
        data.append({"text": sentence[:length]})
    return Dataset.from_list(data)

def prepare_dataset(args, tokenizer):
    """Loads and tokenizes data from various sources."""
    dataset = None
    try:
        if args.dataset_type == "pdf":
            try:
                import pypdf
                if os.path.exists(args.dataset_path):
                    reader = pypdf.PdfReader(args.dataset_path)
                    full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    chunks = [full_text[i:i+250] for i in range(0, len(full_text), 250)]
                    dataset = Dataset.from_list([{"text": c} for c in chunks])
                    logger.info(f"Successfully loaded PDF: {args.dataset_path}")
                else:
                    logger.warning(f"PDF file not found: {args.dataset_path}")
            except ImportError:
                logger.warning("pypdf not installed. Install with 'pip install pypdf'.")

        elif args.dataset_type == "text":
            if os.path.exists(args.dataset_path):
                with open(args.dataset_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                chunks = [full_text[i:i+250] for i in range(0, len(full_text), 250)]
                dataset = Dataset.from_list([{"text": c} for c in chunks])
                logger.info(f"Successfully loaded text file: {args.dataset_path}")
            else:
                logger.warning(f"Text file not found: {args.dataset_path}")

        elif args.dataset_type == "hf":
            try:
                if args.dataset_path == "wikitext":
                    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                else:
                    dataset = load_dataset(args.dataset_path, split="train")
                # Filter out empty or very short strings
                dataset = dataset.filter(lambda x: len(x.get("text", "")) > 10)
                logger.info(f"Successfully loaded HF dataset: {args.dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to load HF dataset {args.dataset_path}: {e}")
    except Exception as e:
        logger.error(f"Error during dataset preparation: {e}")

    if dataset is None or len(dataset) == 0:
        dataset = get_synthetic_dataset()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    return tokenized_ds

# =====================================================================
# 4. ADVANCED UATC v3.3 CLOSED-LOOP TRAINING LOOP
# =====================================================================

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Initializing UATC Session | Model: {args.model_id} | Paradigm: {args.training_paradigm}")

    # --- 4.1 DYNAMIC WORKLOAD & QUANTIZATION CONFIGURATOR ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = None

    if args.training_paradigm == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype if args.training_paradigm != "qlora" else None,
        device_map="auto" if device.type == "cuda" else None,
    )

    if args.training_paradigm == "qlora":
        model = prepare_model_for_kbit_training(model)

    if args.training_paradigm in ["qlora", "peft"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # --- 4.2 DATA & OPTIMIZATION SETUP ---
    tokenized_ds = prepare_dataset(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    controller_cfg = ControllerConfig(
        memory_soft_limit=args.memory_soft_limit,
        memory_hard_limit=args.memory_hard_limit,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        base_pruning_threshold=args.base_pruning_threshold,
    )
    controller = AdaptiveExpertController(controller_cfg)
    metrics = MetricTracker()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # State variables
    current_bs = args.init_batch_size
    current_lr = 1e-4
    current_amp = True
    current_ckpt = False
    current_pruning_ratio = 0.0

    history = {
        "step": [], "vram_raw": [], "vram_kalman": [], "vram_smith": [],
        "bs": [], "loss": [], "pruning_rate": [], "pid_error": [], "pid_out": [],
        "phase": []
    }

    sample_idx = 0
    total_samples = len(tokenized_ds)

    # --- 4.3 CORE TRAINING LOOP ---
    step = 1
    while step <= args.total_steps:
        model.train()

        # Adaptive Checkpointing
        if current_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()

        # Dynamic Batch Slicing
        batch_indices = [(sample_idx + i) % total_samples for i in range(current_bs)]
        sample_idx = (sample_idx + current_bs) % total_samples

        batch_samples = [tokenized_ds[i] for i in batch_indices]
        batch = data_collator(batch_samples).to(device)

        oom_detected = False
        loss_val = 0.0
        grad_norm = 0.0

        try:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=current_amp):
                outputs = model(**batch)
                logits = outputs.logits

                # Causal LM loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

                per_token_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_token_loss = per_token_loss.view(current_bs, -1)

                # Precise per-sample loss (averaging only over non-ignored tokens)
                loss_mask = (shift_labels != -100).float()
                per_sample_loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

                # Per-Sample Loss & Pruning
                filtered_losses, skipped_count, active_thr = controller.pruner.filter_batch_losses(
                    per_sample_loss, controller._phase
                )

                loss = filtered_losses.mean()
                loss_val = loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()

        except torch.cuda.OutOfMemoryError:
            logger.error(f"STRICT OOM EXCEPTION AT STEP {step}. Initiating aggressive recovery...")
            oom_detected = True
            # Flush everything
            if 'outputs' in locals(): del outputs
            if 'logits' in locals(): del logits
            if 'loss' in locals(): del loss
            if 'per_token_loss' in locals(): del per_token_loss
            if 'per_sample_loss' in locals(): del per_sample_loss
            if 'filtered_losses' in locals(): del filtered_losses
            gc.collect()
            torch.cuda.empty_cache()

        # Telemetry Gathering
        vram_pressure = 0.0
        total_vram = 0
        if torch.cuda.is_available():
            vram_pressure = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
            total_vram = torch.cuda.get_device_properties(0).total_memory

        v_loss, var_loss, ema_s, ema_l = metrics.update(loss_val, grad_norm)

        state = ModelState(
            step=step,
            current_batch_size=current_bs,
            current_amp_enabled=current_amp,
            current_checkpointing_enabled=current_ckpt,
            current_pruning_ratio=current_pruning_ratio,
            current_lr=current_lr,
            loss_current=loss_val,
            loss_velocity=v_loss,
            loss_variance=var_loss,
            loss_ema_short=ema_s,
            loss_ema_long=ema_l,
            memory_pressure=vram_pressure,
            grad_norm_ema=metrics.grad_norm_ema,
            oom_detected=oom_detected,
            training_paradigm=args.training_paradigm.upper(),
            lora_rank=args.lora_rank,
            total_vram_bytes=total_vram,
            current_seq_len=batch["input_ids"].size(1)
        )

        # Query Controller
        action = controller.decide(state)

        # Apply Action
        current_bs = action.target_batch_size
        current_lr = action.target_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        current_amp = action.target_amp_enabled
        current_ckpt = action.target_checkpointing_enabled
        current_pruning_ratio = action.target_pruning_ratio

        # Record Telemetry
        history["step"].append(step)
        history["vram_raw"].append(vram_pressure)
        history["vram_kalman"].append(action.metadata.get("mem_pressure_smooth", vram_pressure))
        history["vram_smith"].append(action.metadata.get("smith_predicted_pressure", vram_pressure))
        history["bs"].append(state.current_batch_size)
        history["loss"].append(loss_val)
        history["pruning_rate"].append(current_pruning_ratio)
        history["pid_error"].append(action.metadata.get("pid_error", 0.0))
        history["pid_out"].append(action.metadata.get("pid_out", 0.0))
        history["phase"].append(action.metadata.get("phase", "WARMUP"))

        if step % 10 == 0 or oom_detected:
            logger.info(f"Step {step:03d} | Loss: {loss_val:.4f} | BS: {state.current_batch_size} | VRAM: {vram_pressure:.1%} | Phase: {action.metadata.get('phase')}")

        if action.skip_step:
            logger.info(f"CONTROLLER REQUESTED SKIP at step {step}. Retrying with adjusted parameters...")
            sample_idx = (sample_idx - state.current_batch_size) % total_samples
            gc.collect()
            torch.cuda.empty_cache()
            continue

        step += 1

    # --- 5. FINAL EVALUATION DASHBOARD ---
    if MATPLOTLIB_AVAILABLE:
        logger.info(f"Generating 4-panel dashboard: {args.save_plot_path}")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
        steps = history["step"]

        # Panel 1: VRAM Pressure
        ax1 = axes[0, 0]
        ax1.plot(steps, history["vram_raw"], label="Raw VRAM", alpha=0.3, color="gray")
        ax1.plot(steps, history["vram_kalman"], label="Kalman Filtered", color="blue", lw=2)
        ax1.plot(steps, history["vram_smith"], label="Smith Predicted", linestyle="--", color="cyan")
        ax1.axhline(y=args.memory_soft_limit, color="orange", linestyle="-.", label="Soft Limit")
        ax1.axhline(y=args.memory_hard_limit, color="red", linestyle="-.", label="Hard Limit")
        ax1.set_title("VRAM Pressure Management", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Memory Pressure (ratio)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 2: Batch Size & Phases
        ax2 = axes[0, 1]
        ax2.plot(steps, history["bs"], color="green", lw=2.5, label="Batch Size")
        ax2.set_title("Dynamic Workload & Phase Transitions", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Samples per Step")

        phase_map = {"WARMUP": "gold", "SCALING": "lightgreen", "CONVERGENCE": "skyblue", "RECOVERY": "salmon"}
        for i in range(len(steps)-1):
            ax2.axvspan(steps[i], steps[i+1], color=phase_map.get(history["phase"][i], "white"), alpha=0.2)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Panel 3: Loss & Pruning
        ax3 = axes[1, 0]
        ax3.plot(steps, history["loss"], color="purple", lw=1.5, label="CE Loss")
        ax3.set_ylabel("Cross-Entropy Loss")
        ax3.set_title("Loss Trajectory & Data Pruning", fontsize=14, fontweight='bold')

        ax3b = ax3.twinx()
        ax3b.bar(steps, history["pruning_rate"], alpha=0.25, color="red", width=1.0, label="Pruning Ratio")
        ax3b.set_ylabel("Pruning Active Ratio")
        ax3.grid(True, alpha=0.3)

        # Merge legends
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper right')

        # Panel 4: PID Feedback
        ax4 = axes[1, 1]
        ax4.plot(steps, history["pid_error"], label="PID Error", color="darkred", alpha=0.8)
        ax4.plot(steps, history["pid_out"], label="PID Regulatory Output", color="darkblue", lw=2)
        ax4.set_title("Control System Regulatory Feedback", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Controller Signal")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(args.save_plot_path)
        logger.info("Scientific Dashboard successfully exported.")

        # Colab download trigger
        try:
            from google.colab import files
            files.download(args.save_plot_path)
        except:
            pass
    else:
        logger.warning("Matplotlib not detected. Dashboard visualization skipped.")

if __name__ == "__main__":
    main()

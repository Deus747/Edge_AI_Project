#!/usr/bin/env python3
# ============================================================
# train_baseline.py — Baseline student pretraining (CE only)
#
# Trains MobileNetV4-S + LR-ASPP with cross-entropy loss only.
# No teacher, no distillation.  Provides a reference point to
# measure how much each KD strategy contributes on top of
# plain supervised training.
#
# Saves best checkpoint to:
#   checkpoints/<LEVEL>/student_baseline_best.pth  (canonical)
#
# Usage:
#   python train_baseline.py
#   IDD_LABEL_LEVEL=Label3ID python train_baseline.py
# ============================================================

import os
import sys
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import config
from dataset import get_loaders
from models.student import build_student
from utils import (
    validate, save_checkpoint, load_checkpoint,
    save_latest, load_latest,
    save_prediction_grid, save_loss_curve,
    save_miou_curve, save_per_class_bar,
)

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if config.DEVICE == 'cuda':
    torch.cuda.manual_seed_all(config.SEED)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
print(f"Label level : {config.LABEL_LEVEL}  ({config.NUM_CLASSES} classes)")
print("Loading data ...")
train_loader, val_loader = get_loaders()

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
print("\nBuilding baseline student ...")
student = build_student()
n = sum(p.numel() for p in student.parameters()) / 1e6
print(f"  {student.BACKBONE_NAME} + LR-ASPP  ({n:.1f}M params)")

# ------------------------------------------------------------------
# Optimiser & scheduler
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=config.BASELINE_LR,
    weight_decay=config.BASELINE_WD,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.BASELINE_EPOCHS, eta_min=1e-6,
)
scaler = torch.amp.GradScaler(
    'cuda', enabled=config.AMP and config.DEVICE == 'cuda',
)

# ------------------------------------------------------------------
# Resume logic
# ------------------------------------------------------------------
LATEST_CKPT = os.path.join(config.CKPT_DIR, 'baseline_latest.pth')
best_miou   = 0.0
best_ckpt   = None
start_epoch = 1

if os.path.exists(LATEST_CKPT):
    print(f"\nFound latest checkpoint — resuming ...")
    start_epoch, best_miou = load_latest(
        student, optimizer, scheduler, LATEST_CKPT, config.DEVICE,
    )
else:
    print(f"\nNo checkpoint found — starting from scratch.")

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
train_log  = []
miou_log   = []

remaining = config.BASELINE_EPOCHS - start_epoch + 1
print(f"\nBaseline student training: epochs {start_epoch}–{config.BASELINE_EPOCHS} "
      f"({remaining} remaining) ...")

for epoch in range(start_epoch, config.BASELINE_EPOCHS + 1):

    student.train()
    epoch_loss = 0.0
    n_batches  = 0

    pbar = tqdm(train_loader,
                desc=f"Baseline Epoch {epoch}/{config.BASELINE_EPOCHS}")
    for imgs, masks in pbar:
        imgs  = imgs.to(config.DEVICE, non_blocking=True)
        masks = masks.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(
                'cuda', enabled=config.AMP and config.DEVICE == 'cuda'):
            out  = student(imgs)
            loss = F.cross_entropy(
                out['logits'], masks,
                ignore_index=config.IGNORE_INDEX,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({'ce': f"{epoch_loss/n_batches:.4f}"})

    scheduler.step()

    avg_loss = epoch_loss / n_batches
    train_log.append({'epoch': epoch, 'ce': avg_loss})
    print(f"Baseline Epoch {epoch:03d} | ce={avg_loss:.4f}")

    val_miou, _ = validate(
        student, val_loader,
        config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
        desc='Val',
    )
    miou_log.append((epoch, val_miou))

    # Save latest checkpoint every epoch (enables resume)
    save_latest(student, optimizer, scheduler, epoch,
                LATEST_CKPT, best_miou=best_miou)

    if val_miou > best_miou:
        best_miou = val_miou
        best_ckpt = save_checkpoint(
            student, optimizer, scheduler,
            epoch, val_miou, tag='student_baseline_best',
        )
        # Save canonical copy for easy reference
        canonical = os.path.join(config.CKPT_DIR, 'student_baseline_best.pth')
        shutil.copy(best_ckpt, canonical)
        print(f"  → Copied to {canonical}")

# ------------------------------------------------------------------
# Final validation on best checkpoint
# ------------------------------------------------------------------
print(f"\nBaseline training done. Best Val mIoU: {best_miou*100:.2f}%")
print("\nLoading best checkpoint for final evaluation ...")
load_checkpoint(student, os.path.join(config.CKPT_DIR, 'student_baseline_best.pth'),
                config.DEVICE)

_, val_per_class = validate(
    student, val_loader,
    config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
    desc='Val (best ckpt)',
)

# ------------------------------------------------------------------
# Save plots
# ------------------------------------------------------------------
print("\nSaving plots ...")

save_loss_curve(
    train_log,
    os.path.join(config.PLOT_DIR, 'baseline_loss_curve.png'),
    tag='Baseline Student',
)
save_miou_curve(
    miou_log,
    os.path.join(config.PLOT_DIR, 'baseline_miou_curve.png'),
    tag='Baseline Student',
)
save_per_class_bar(
    val_per_class,
    os.path.join(config.PLOT_DIR, 'baseline_per_class_val.png'),
    tag='Baseline Student (Val)',
)
save_prediction_grid(
    student, val_loader, config.DEVICE,
    os.path.join(config.PLOT_DIR, 'baseline_predictions_val.png'),
    title='Baseline Student — Validation Predictions',
)

print("\nBaseline student training complete.")
print(f"  Checkpoint : {best_ckpt}")
print(f"  Plots      : {config.PLOT_DIR}")

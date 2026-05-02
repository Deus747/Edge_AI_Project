#!/usr/bin/env python3
# ============================================================
# train_teacher.py — Teacher pretraining on IDD
#
# Trains MobileNetV4-L + DeepLabV3+ with CE loss.
# Saves best checkpoint to config.TEACHER_CKPT.
# Saves plots to config.PLOT_DIR.
#
# Usage:
#   python train_teacher.py
# ============================================================

import os
import sys
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make project root importable
sys.path.insert(0, os.path.dirname(__file__))

import config
from dataset import get_loaders
from models.teacher import build_teacher
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
print("Loading data ...")
train_loader, val_loader = get_loaders()

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
print("\nBuilding teacher ...")
teacher = build_teacher(freeze=False)   # unfreeze for pretraining

# ------------------------------------------------------------------
# Optimiser & scheduler
# ------------------------------------------------------------------
optimizer = torch.optim.AdamW(
    teacher.parameters(),
    lr=config.TEACHER_LR,
    weight_decay=config.TEACHER_WD,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.TEACHER_EPOCHS, eta_min=1e-6,
)
scaler = torch.amp.GradScaler(
    'cuda', enabled=config.AMP and config.DEVICE == 'cuda',
)

# ------------------------------------------------------------------
# Resume logic
# ------------------------------------------------------------------
LATEST_CKPT = os.path.join(config.CKPT_DIR, 'teacher_latest.pth')
best_miou   = 0.0
start_epoch = 1

if os.path.exists(LATEST_CKPT):
    print(f"\nFound latest checkpoint — resuming ...")
    start_epoch, best_miou = load_latest(
        teacher, optimizer, scheduler, LATEST_CKPT, config.DEVICE,
    )
else:
    print(f"\nNo checkpoint found — starting from scratch.")

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
train_log  = []
miou_log   = []

remaining = config.TEACHER_EPOCHS - start_epoch + 1
print(f"\nTeacher pretraining: epochs {start_epoch}–{config.TEACHER_EPOCHS} "
      f"({remaining} remaining) ...")

for epoch in range(start_epoch, config.TEACHER_EPOCHS + 1):

    teacher.train()
    epoch_loss = 0.0
    n_batches  = 0

    pbar = tqdm(train_loader,
                desc=f"Teacher Epoch {epoch}/{config.TEACHER_EPOCHS}")
    for imgs, masks in pbar:
        imgs  = imgs.to(config.DEVICE, non_blocking=True)
        masks = masks.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(
                'cuda', enabled=config.AMP and config.DEVICE == 'cuda'):
            out  = teacher(imgs)
            loss = F.cross_entropy(
                out['logits'], masks,
                ignore_index=config.IGNORE_INDEX,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            teacher.parameters(), max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({'ce': f"{epoch_loss/n_batches:.4f}"})

    scheduler.step()

    avg_loss = epoch_loss / n_batches
    train_log.append({'epoch': epoch, 'ce': avg_loss})
    print(f"Teacher Epoch {epoch:03d} | ce={avg_loss:.4f}")

    # Validate
    val_miou, val_per_class = validate(
        teacher, val_loader,
        config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
        desc='Val',
    )
    miou_log.append((epoch, val_miou))

    # Save latest checkpoint every epoch (enables resume)
    save_latest(teacher, optimizer, scheduler, epoch,
                LATEST_CKPT, best_miou=best_miou)

    # Save best checkpoint
    if val_miou > best_miou:
        best_miou = val_miou
        save_checkpoint(
            teacher, optimizer, scheduler,
            epoch, val_miou, tag='teacher_best',
        )
        # Also save to the canonical TEACHER_CKPT path for KD scripts
        latest = os.path.join(
            config.CKPT_DIR,
            f"teacher_best_ep{epoch:03d}_miou{val_miou*100:.1f}.pth",
        )
        shutil.copy(latest, config.TEACHER_CKPT)
        print(f"  → Copied to {config.TEACHER_CKPT}")

# ------------------------------------------------------------------
# Re-freeze teacher
# ------------------------------------------------------------------
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
print(f"\nPretraining done. Best Val mIoU: {best_miou*100:.2f}%")

# ------------------------------------------------------------------
# Final validation on best checkpoint
# ------------------------------------------------------------------
print("\nLoading best checkpoint for final evaluation ...")
load_checkpoint(teacher, config.TEACHER_CKPT, config.DEVICE)
teacher.eval()

_, val_per_class = validate(
    teacher, val_loader,
    config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
    desc='Val (best ckpt)',
)

# ------------------------------------------------------------------
# Save plots
# ------------------------------------------------------------------
print("\nSaving plots ...")

save_loss_curve(
    train_log,
    os.path.join(config.PLOT_DIR, 'teacher_loss_curve.png'),
    tag='Teacher',
)
save_miou_curve(
    miou_log,
    os.path.join(config.PLOT_DIR, 'teacher_miou_curve.png'),
    tag='Teacher',
)
save_per_class_bar(
    val_per_class,
    os.path.join(config.PLOT_DIR, 'teacher_per_class_val.png'),
    tag='Teacher (Val)',
)
save_prediction_grid(
    teacher, val_loader, config.DEVICE,
    os.path.join(config.PLOT_DIR, 'teacher_predictions_val.png'),
    title='Teacher — Validation Predictions',
)

print("\nTeacher training complete.")
print(f"  Checkpoint : {config.TEACHER_CKPT}")
print(f"  Plots      : {config.PLOT_DIR}")

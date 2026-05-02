#!/usr/bin/env python3
# ============================================================
# train_kd_base.py — Shared KD training logic
# Not called directly. Imported by train_logit_kd.py.
# The entry script calls run_kd() with the logit-KD loss module.
# ============================================================

import os
import sys
import random
import shutil
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import config
from dataset import get_loaders
from models.teacher import build_teacher
from models.student import build_student
from utils import (
    validate, save_checkpoint, load_checkpoint,
    save_latest, load_latest,
    save_prediction_grid, save_loss_curve,
    save_miou_curve, save_per_class_bar,
)


def run_kd(loss_module, strategy_tag):
    """
    Full KD training pipeline.

    Args:
        loss_module : imported losses.loss_X module
                      must expose build_loss()
        strategy_tag: string label used for filenames e.g. 'logit_kd'
    """

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
    # Teacher — load pretrained weights, freeze
    # ------------------------------------------------------------------
    print("\nLoading teacher ...")
    assert os.path.exists(config.TEACHER_CKPT), (
        f"Teacher checkpoint not found: {config.TEACHER_CKPT}\n"
        f"Run train_teacher.py first."
    )
    teacher = build_teacher(freeze=True)
    load_checkpoint(teacher, config.TEACHER_CKPT, config.DEVICE)
    teacher.eval()

    # ------------------------------------------------------------------
    # Student
    # ------------------------------------------------------------------
    print("\nBuilding student ...")
    student = build_student()

    # ------------------------------------------------------------------
    # KD Loss
    # ------------------------------------------------------------------
    print("\nBuilding KD loss ...")
    kd_criterion = loss_module.build_loss()

    # ------------------------------------------------------------------
    # Optimiser — student params + projector params (if any)
    # ------------------------------------------------------------------
    trainable_params = (
        list(student.parameters()) +
        list(kd_criterion.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.KD_LR,
        weight_decay=config.KD_WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.KD_EPOCHS, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=config.AMP and config.DEVICE == 'cuda',
    )

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    LATEST_CKPT   = os.path.join(config.CKPT_DIR, f'{strategy_tag}_latest.pth')
    CANONICAL_BEST = os.path.join(config.CKPT_DIR, f'{strategy_tag}_best.pth')
    best_miou   = 0.0
    best_ckpt   = CANONICAL_BEST if os.path.exists(CANONICAL_BEST) else None
    start_epoch = 1

    if os.path.exists(LATEST_CKPT):
        print(f"\nFound latest checkpoint — resuming ...")
        start_epoch, best_miou = load_latest(
            student, optimizer, scheduler, LATEST_CKPT, config.DEVICE,
            criterion=kd_criterion,
        )
    else:
        print(f"\nNo checkpoint found — starting from scratch.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_log  = []
    miou_log   = []

    remaining = config.KD_EPOCHS - start_epoch + 1
    print(f"\nKD training [{strategy_tag}]: epochs {start_epoch}–{config.KD_EPOCHS} "
          f"({remaining} remaining) ...")

    for epoch in range(start_epoch, config.KD_EPOCHS + 1):

        student.train()
        teacher.eval()
        kd_criterion.train()

        epoch_losses = {}
        n_batches    = 0

        pbar = tqdm(train_loader,
                    desc=f"[{strategy_tag}] Epoch {epoch}/{config.KD_EPOCHS}")

        for imgs, masks in pbar:
            imgs  = imgs.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(
                    'cuda', enabled=config.AMP and config.DEVICE == 'cuda'):
                with torch.no_grad():
                    t_out = teacher(imgs)
                s_out = student(imgs)
                loss, loss_dict = kd_criterion(s_out, t_out, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1
            pbar.set_postfix({k: f"{v/n_batches:.4f}"
                              for k, v in epoch_losses.items()})

        scheduler.step()

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        avg['epoch'] = epoch
        train_log.append(avg)
        print(f"[{strategy_tag}] Epoch {epoch:03d} | "
              + " | ".join(f"{k}={v:.4f}"
                           for k, v in avg.items() if k != 'epoch'))

        # Validate
        val_miou, _ = validate(
            student, val_loader,
            config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
            desc=f'[{strategy_tag}] Val',
        )
        miou_log.append((epoch, val_miou))

        # Save latest checkpoint every epoch (enables resume)
        save_latest(student, optimizer, scheduler, epoch,
                    LATEST_CKPT, best_miou=best_miou,
                    criterion=kd_criterion)

        # Save best checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            best_ckpt = save_checkpoint(
                student, optimizer, scheduler,
                epoch, val_miou,
                tag=f'{strategy_tag}_best',
                extra={'strategy': strategy_tag},
            )
            shutil.copy(best_ckpt, CANONICAL_BEST)

    # ------------------------------------------------------------------
    # Final validation on best checkpoint
    # ------------------------------------------------------------------
    print(f"\n[{strategy_tag}] Training done. "
          f"Best Val mIoU: {best_miou*100:.2f}%")
    print(f"Loading best checkpoint for final evaluation ...")
    load_checkpoint(student, CANONICAL_BEST, config.DEVICE)

    print(f"\nTeacher — Val:")
    _, t_val_per = validate(
        teacher, val_loader,
        config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
        desc='Teacher Val',
    )
    print(f"\n[{strategy_tag}] Student — Val:")
    _, s_val_per = validate(
        student, val_loader,
        config.NUM_CLASSES, config.IGNORE_INDEX, config.DEVICE,
        desc='Student Val',
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\n── Per-class IoU Comparison [{strategy_tag}] ──")
    print(f"{'Class':<26} {'Teacher':>10} {'Student':>10} {'Gap':>10}")
    print("-" * 58)
    for name, t_iou, s_iou in zip(
            config.CLASS_NAMES, t_val_per, s_val_per):
        gap  = s_iou - t_iou
        flag = ' ↑' if gap > 0 else (' ↓' if gap < -0.02 else '')
        print(f"{name:<26} {t_iou*100:>9.2f}% "
              f"{s_iou*100:>9.2f}% {gap*100:>+9.2f}%{flag}")
    print("-" * 58)

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    print("\nSaving plots ...")

    save_loss_curve(
        train_log,
        os.path.join(config.PLOT_DIR,
                     f'{strategy_tag}_loss_curve.png'),
        tag=strategy_tag,
    )
    save_miou_curve(
        miou_log,
        os.path.join(config.PLOT_DIR,
                     f'{strategy_tag}_miou_curve.png'),
        tag=strategy_tag,
    )
    save_per_class_bar(
        s_val_per,
        os.path.join(config.PLOT_DIR,
                     f'{strategy_tag}_per_class_val.png'),
        tag=f'Student [{strategy_tag}] Val',
        reference=t_val_per,
        ref_label='Teacher',
    )
    save_prediction_grid(
        student, val_loader, config.DEVICE,
        os.path.join(config.PLOT_DIR,
                     f'{strategy_tag}_predictions_val.png'),
        title=f'Student [{strategy_tag}] vs Teacher — Validation',
        teacher=teacher,
    )

    print(f"\n[{strategy_tag}] Done.")
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Plots           : {config.PLOT_DIR}")

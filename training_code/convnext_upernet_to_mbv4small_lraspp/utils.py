# ============================================================
# utils.py — validate, checkpoint, visualisation, plotting
# ============================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

import config


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate(model, loader, num_classes, ignore_index, device,
             desc='Validating'):
    """
    Returns (mIoU scalar, per-class IoU numpy array).
    Prints per-class breakdown.
    """
    model.eval()
    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index,
        average='macro',
    ).to(device)
    per_class_metric = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index,
        average='none',
    ).to(device)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=desc, leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast(
                    'cuda', enabled=config.AMP and device == 'cuda'):
                out = model(imgs)
            preds = out['logits'].argmax(dim=1)
            miou_metric.update(preds, masks)
            per_class_metric.update(preds, masks)

    miou      = miou_metric.compute().item()
    per_class = per_class_metric.compute().cpu().numpy()

    print(f"\n{desc} mIoU: {miou*100:.2f}%")
    for name, iou in zip(config.CLASS_NAMES, per_class):
        print(f"  {name:<24}: {iou*100:.2f}%")
    return miou, per_class


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch,
                    miou, tag, extra=None):
    """Save model checkpoint to CKPT_DIR."""
    path = os.path.join(
        config.CKPT_DIR,
        f"{tag}_ep{epoch:03d}_miou{miou*100:.1f}.pth",
    )
    payload = {
        'epoch'      : epoch,
        'miou'       : miou,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Checkpoint saved → {path}")
    return path


def load_checkpoint(model, path, device,
                    optimizer=None, scheduler=None):
    """Load model weights (and optionally optimizer/scheduler)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler and 'sched_state' in ckpt:
        scheduler.load_state_dict(ckpt['sched_state'])
    print(f"Loaded checkpoint: {path}  "
          f"epoch={ckpt['epoch']}  mIoU={ckpt['miou']*100:.2f}%")
    return ckpt['epoch'], ckpt['miou']


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def mask_to_rgb(mask_np):
    """Convert 2D class index mask to RGB image."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in config.PALETTE.items():
        rgb[mask_np == cid] = col
    rgb[mask_np == config.IGNORE_INDEX] = (128, 128, 128)
    return rgb


def denorm(tensor):
    """Reverse ImageNet normalisation for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    return (img * std + mean).clip(0, 1)


def per_sample_miou(pred, gt_mask, num_classes, ignore_index):
    """Quick per-image mIoU scalar for plot labels."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c   = (gt_mask == c) & (gt_mask != ignore_index)
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) * 100 if ious else 0.0


def save_prediction_grid(model, loader, device, save_path,
                         title='Predictions', n_show=4,
                         teacher=None):
    """
    Save a prediction grid PNG.
    If teacher is provided: 4-col grid (image | gt | teacher | model).
    Otherwise: 3-col grid (image | gt | model).
    """
    model.eval()
    if teacher is not None:
        teacher.eval()

    imgs, masks = next(iter(loader))
    imgs  = imgs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        with torch.amp.autocast(
                'cuda', enabled=config.AMP and device == 'cuda'):
            s_out = model(imgs)
            t_out = teacher(imgs) if teacher is not None else None

    s_preds = s_out['logits'].argmax(dim=1).cpu().numpy()
    t_preds = t_out['logits'].argmax(dim=1).cpu().numpy() \
              if t_out is not None else None
    gt      = masks.cpu().numpy()

    n_show = min(n_show, imgs.shape[0])
    n_cols = 4 if teacher is not None else 3
    fig, axes = plt.subplots(n_show, n_cols,
                             figsize=(5 * n_cols, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    col_titles = (['Input', 'Ground Truth', 'Teacher', 'Student (KD)']
                  if teacher is not None
                  else ['Input', 'Ground Truth', 'Prediction'])
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=11, fontweight='bold')

    for i in range(n_show):
        axes[i, 0].imshow(denorm(imgs[i].cpu()))
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask_to_rgb(gt[i]))
        axes[i, 1].axis('off')

        if teacher is not None:
            t_miou = per_sample_miou(t_preds[i], gt[i],
                                     config.NUM_CLASSES,
                                     config.IGNORE_INDEX)
            s_miou = per_sample_miou(s_preds[i], gt[i],
                                     config.NUM_CLASSES,
                                     config.IGNORE_INDEX)
            axes[i, 2].imshow(mask_to_rgb(t_preds[i]))
            axes[i, 2].set_xlabel(f"mIoU: {t_miou:.1f}%", fontsize=9)
            axes[i, 2].axis('off')
            axes[i, 3].imshow(mask_to_rgb(s_preds[i]))
            axes[i, 3].set_xlabel(f"mIoU: {s_miou:.1f}%", fontsize=9)
            axes[i, 3].axis('off')
        else:
            p_miou = per_sample_miou(s_preds[i], gt[i],
                                     config.NUM_CLASSES,
                                     config.IGNORE_INDEX)
            axes[i, 2].imshow(mask_to_rgb(s_preds[i]))
            axes[i, 2].set_xlabel(f"mIoU: {p_miou:.1f}%", fontsize=9)
            axes[i, 2].axis('off')

    patches = [
        mpatches.Patch(color=[c/255 for c in col],
                       label=config.CLASS_NAMES[cid])
        for cid, col in config.PALETTE.items()
    ]
    fig.legend(handles=patches, loc='lower center',
               ncol=min(7, config.NUM_CLASSES),
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Prediction grid saved → {save_path}")


# ------------------------------------------------------------------
# Training curve plots
# ------------------------------------------------------------------

def save_loss_curve(train_log, save_path, tag=''):
    """
    Save loss curves for all keys in train_log dicts.
    train_log: list of dicts with keys 'epoch', 'ce', 'kl', etc.
    """
    if not train_log:
        return
    epochs = [d['epoch'] for d in train_log]
    keys   = [k for k in train_log[0] if k != 'epoch']

    fig, axes = plt.subplots(1, len(keys),
                             figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        vals = [d[key] for d in train_log]
        ax.plot(epochs, vals, marker='o', markersize=3, linewidth=1.5)
        ax.set_title(f"{key} loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training curves — {tag}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss curve saved → {save_path}")


def save_miou_curve(miou_log, save_path, tag=''):
    """
    Save mIoU over epochs.
    miou_log: list of (epoch, miou) tuples.
    """
    if not miou_log:
        return
    epochs = [e for e, _ in miou_log]
    mious  = [m * 100 for _, m in miou_log]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mious, marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Val mIoU (%)')
    plt.title(f"Validation mIoU — {tag}", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  mIoU curve saved → {save_path}")


def save_per_class_bar(per_class, save_path, tag='', reference=None,
                       ref_label='Teacher'):
    """
    Save per-class IoU bar chart.
    Optionally overlays reference (teacher) bars.
    """
    n   = len(config.CLASS_NAMES)
    x   = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))

    if reference is not None:
        ax.bar(x - 0.2, reference * 100, 0.4,
               label=ref_label, alpha=0.7, color='steelblue')
        ax.bar(x + 0.2, per_class * 100, 0.4,
               label=tag, alpha=0.7, color='tomato')
        ax.legend()
    else:
        ax.bar(x, per_class * 100, color='steelblue', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45,
                       ha='right', fontsize=8)
    ax.set_ylabel('IoU (%)')
    ax.set_ylim(0, 100)
    ax.set_title(f"Per-class IoU — {tag}", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class bar saved → {save_path}")


# ------------------------------------------------------------------
# Smart resume checkpoints
# ------------------------------------------------------------------

def save_latest(model, optimizer, scheduler, epoch, path,
                best_miou=0.0, criterion=None):
    """
    Overwrite a single 'latest' checkpoint at a fixed path each epoch.
    Stores best_miou so resume can restore the correct tracking value.
    Pass criterion to also save projector/loss module weights (KD scripts).
    """
    payload = {
        'epoch'      : epoch,
        'best_miou'  : best_miou,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict(),
    }
    if criterion is not None:
        payload['criterion_state'] = criterion.state_dict()
    torch.save(payload, path)


def load_latest(model, optimizer, scheduler, path, device,
                criterion=None):
    """
    Load a latest checkpoint saved by save_latest.
    Returns (start_epoch, best_miou) where start_epoch = last_epoch + 1.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])
    if criterion is not None and 'criterion_state' in ckpt:
        criterion.load_state_dict(ckpt['criterion_state'])
    last_epoch = ckpt['epoch']
    best_miou  = ckpt.get('best_miou', 0.0)
    print(f"  Resuming from {os.path.basename(path)} "
          f"(last epoch={last_epoch}, best mIoU={best_miou*100:.2f}%)")
    return last_epoch + 1, best_miou

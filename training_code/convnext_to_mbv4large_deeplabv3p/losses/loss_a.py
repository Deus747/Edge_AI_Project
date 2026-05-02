# ============================================================
# losses/loss_a.py — Strategy A: Logit KD only
#
# L = CE(z_S, y) + alpha * KL(z_T/tau || z_S/tau)
#
# Suitable for architecturally incompatible teacher/student
# pairs where feature-level alignment is not possible.
# Pair A: ResNet-50 + DeepLabV3+ → MobileNetV2 + DeepLabV3
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def seg_ce_loss(logits, targets, ignore_index):
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


class KDLoss(nn.Module):
    """
    Response-based KD: CE + KL divergence on soft logits.
    Spatial dims flattened before KL to avoid batchmean scaling issues.
    """
    def __init__(self, temperature=None, alpha=None,
                 ignore_index=None, **kwargs):
        super().__init__()
        self.tau    = temperature  or config.TEMPERATURE
        self.alpha  = alpha        or config.ALPHA
        self.ignore = ignore_index or config.IGNORE_INDEX

    def forward(self, s_out, t_out, targets):
        # Term 1: CE
        ce = seg_ce_loss(s_out['logits'], targets, self.ignore)

        # Term 2: Logit KD — flatten spatial before KL and ignore
        # pixels marked with ignore_index in the target mask.
        B, C, H, W = s_out['logits'].shape
        s_flat = s_out['logits'].permute(0, 2, 3, 1).reshape(-1, C)
        t_flat = t_out['logits'].permute(0, 2, 3, 1).reshape(-1, C)
        valid = targets.reshape(-1) != self.ignore
        if valid.any():
            s_valid = s_flat[valid]
            t_valid = t_flat[valid]
            kl = F.kl_div(
                F.log_softmax(s_valid / self.tau, dim=1),
                F.softmax(t_valid.detach() / self.tau, dim=1),
                reduction='batchmean',
            ) * self.tau ** 2
        else:
            kl = s_out['logits'].new_tensor(0.0)

        loss = ce + self.alpha * kl
        return loss, {
            'ce'   : ce.item(),
            'kl'   : kl.item(),
            'total': loss.item(),
        }


def build_loss():
    criterion = KDLoss()
    print(f"KD Loss: Strategy A — Logit KD only")
    print(f"  alpha={criterion.alpha}  tau={criterion.tau}")
    return criterion

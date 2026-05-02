# ============================================================
# models/probe.py — Verify actual backbone channel dims
# Run before training if you change the backbone.
# timm features_only stage output channels are NOT the same
# as the paper's internal widths — always probe first.
#
# Usage: python models/probe.py
# ============================================================

import torch
import timm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def probe(name, label):
    bb = timm.create_model(
        name, pretrained=False,
        features_only=True, out_indices=(0, 1, 2, 3),
    ).to(config.DEVICE)
    with torch.no_grad():
        dummy = torch.randn(1, 3, *config.IMG_SIZE).to(config.DEVICE)
        feats = bb(dummy)
    channels = [f.shape[1] for f in feats]
    spatial  = [tuple(f.shape[-2:]) for f in feats]
    print(f"\n{label} ({name})")
    for i, (ch, sp) in enumerate(zip(channels, spatial)):
        print(f"  Stage {i+1}: {ch:>4} ch  {sp[0]}x{sp[1]}")
    print(f"  -> set config.{label.upper()}_BB_CHANNELS = {channels}")
    del bb, dummy, feats


if __name__ == '__main__':
    probe('convnext_base', 'teacher')
    probe('mobilenetv4_conv_small.e2400_r224_in1k', 'student')

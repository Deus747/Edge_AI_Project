#!/usr/bin/env python3
# ============================================================
# train_logit_kd.py - Logit KD training
#
# L = CE + alpha * KL(z_T/tau || z_S/tau)
#
# Usage:
#   python train_logit_kd.py
# ============================================================

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from losses import loss_a
from train_kd_base import run_kd

if __name__ == '__main__':
    run_kd(loss_module=loss_a, strategy_tag='logit_kd')

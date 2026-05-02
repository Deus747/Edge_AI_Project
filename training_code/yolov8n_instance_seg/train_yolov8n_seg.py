#!/usr/bin/env python3
import os
import random

import numpy as np
import torch
from ultralytics import YOLO

import config
from scripts.prepare_dataset_bridge import main as prepare_bridge


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(42)

    print("Preparing dataset bridge...")
    prepare_bridge()

    print("Starting YOLOv8n-seg training...")
    print(f"  Data YAML : {config.DATA_YAML}")
    print(f"  Runs dir  : {config.RUNS_DIR}")
    print(f"  Device    : {config.DEVICE}")

    model = YOLO(config.MODEL)
    model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        workers=config.WORKERS,
        device=config.DEVICE,
        project=config.RUNS_DIR,
        name=config.PROJECT_NAME,
        exist_ok=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()

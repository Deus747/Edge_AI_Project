#!/usr/bin/env python3

import argparse
import glob
import os
import pathlib

import numpy as np
from PIL import Image


IMGSZ = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "calibs"),
        help="Directory containing source calibration images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parent / "calibs_npy"),
        help="Directory where .npy calibration tensors will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PREPARING YOLOV8N SEG CALIBRATION DATA")
    print("=" * 60)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"no calibration images found in {args.input_dir}")

    print(f"\nFound {len(image_paths)} images in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Processing...")
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMGSZ, IMGSZ), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32)
        np.save(os.path.join(args.output_dir, f"calib_{idx:04d}.npy"), img_array)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(image_paths)}")

    print(f"\nSaved {len(image_paths)} calibration tensors to {args.output_dir}")


if __name__ == "__main__":
    main()

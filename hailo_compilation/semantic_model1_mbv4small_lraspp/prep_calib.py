#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBS_DIR = SCRIPT_DIR.parent / "calibs"
OUTPUT_DIR = SCRIPT_DIR / "calibs_npy"
IMAGE_SIZE = (512, 512)
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images():
    for path in sorted(CALIBS_DIR.iterdir()):
        if path.suffix.lower() in VALID_SUFFIXES and path.is_file():
            yield path


def main():
    if not CALIBS_DIR.exists():
        raise FileNotFoundError(f"Calibration image directory not found: {CALIBS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in iter_images():
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        out_path = OUTPUT_DIR / f"{image_path.stem}.npy"
        np.save(out_path, array)
        count += 1

    print(f"wrote {count} calibration tensors to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

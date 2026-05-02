# ============================================================
# dataset.py — Dataset, label maps, transforms, dataloaders
# ============================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

MASK_ENCODING_MODE = os.environ.get('IDD_MASK_ENCODING', 'auto')

# ------------------------------------------------------------------
# Label maps — raw pixel id → contiguous training id
# ------------------------------------------------------------------

LABEL1ID_MAP = {
    0: 0,   # road
    1: 1,   # sky
    2: 2,   # vegetation
    3: 3,   # building
    4: 4,   # vehicle
    5: 5,   # person
    6: 6,   # background
}

LABEL2ID_MAP = {
    # Maps raw IDD label id (label.py `id` field) → contiguous L2 training id
    # 16 classes (ids 0-15) matching the level2Id column in label.py
    0:  0,   # road
    1:  1,   # parking            → drivable-other
    2:  1,   # drivable fallback  → drivable-other
    3:  2,   # sidewalk
    4:  3,   # rail track         → non-drivable-other
    5:  3,   # non-drivable fallback → non-drivable-other
    6:  4,   # person
    7:  4,   # animal             → person
    8:  5,   # rider
    9:  6,   # motorcycle         → 2-wheeler
    10: 6,   # bicycle            → 2-wheeler
    11: 7,   # autorickshaw       → small-vehicle (merged with car at L2)
    12: 7,   # car                → small-vehicle
    13: 8,   # truck              → large-vehicle
    14: 8,   # bus                → large-vehicle
    15: 8,   # caravan            → large-vehicle
    16: 8,   # trailer            → large-vehicle
    17: 8,   # train              → large-vehicle
    18: 8,   # vehicle fallback   → large-vehicle
    19: 9,   # curb               → barrier-solid
    20: 9,   # wall               → barrier-solid
    21: 10,  # fence              → barrier-open
    22: 10,  # guard rail         → barrier-open
    23: 11,  # billboard          → structures-sign
    24: 11,  # traffic sign       → structures-sign
    25: 11,  # traffic light      → structures-sign
    26: 12,  # pole               → structures-pole
    27: 12,  # polegroup          → structures-pole
    28: 12,  # obs-str-bar-fallback → structures-pole
    29: 13,  # building           → construction
    30: 13,  # bridge             → construction
    31: 13,  # tunnel             → construction
    32: 14,  # vegetation
    33: 15,  # sky
    34: 15,  # fallback background → sky
    # 35-39: ignore classes — not in map → IGNORE_INDEX
}

LABEL3ID_MAP = {
    0 : 0,   # road
    1 : 1,   # parking
    2 : 1,   # drivable fallback       → same l3 as parking
    3 : 2,   # sidewalk
    4 : 3,   # rail track
    5 : 3,   # non-drivable fallback   → same l3 as rail track
    6 : 4,   # person
    7 : 4,   # animal                  → same l3 as person
    8 : 5,   # rider
    9 : 6,   # motorcycle
    10: 7,   # bicycle
    11: 8,   # autorickshaw
    12: 9,   # car
    13: 10,  # truck
    14: 11,  # bus
    15: 12,  # caravan
    16: 12,  # trailer                 → same l3 as caravan
    17: 12,  # train                   → same l3 as caravan
    18: 12,  # vehicle fallback        → same l3 as caravan
    19: 13,  # curb
    20: 14,  # wall
    21: 15,  # fence
    22: 16,  # guard rail
    23: 17,  # billboard
    24: 18,  # traffic sign
    25: 19,  # traffic light
    26: 20,  # pole
    27: 20,  # polegroup               → same l3 as pole
    28: 21,  # obs-str-bar-fallback
    29: 22,  # building
    30: 23,  # bridge
    31: 23,  # tunnel                  → same l3 as bridge
    32: 24,  # vegetation
    33: 25,  # sky
    34: 25,  # fallback background     → same l3 as sky
    # 35-39: ignore classes — not in map → IGNORE_INDEX
}

LABEL_MAPS = {
    'Label1ID': LABEL1ID_MAP,
    'Label2ID': LABEL2ID_MAP,
    'Label3ID': LABEL3ID_MAP,
}


def remap_mask(mask_np, label_map, ignore_index=255):
    """Map raw pixel values to contiguous training ids."""
    out = np.full_like(mask_np, ignore_index, dtype=np.uint8)
    for raw_id, train_id in label_map.items():
        out[mask_np == raw_id] = train_id
    return out


def _mask_already_train_ids(mask_np, num_classes, ignore_index):
    """Return True if mask is already contiguous train ids + ignore."""
    uniq = np.unique(mask_np)
    valid = set(range(num_classes))
    valid.add(ignore_index)
    return set(int(v) for v in uniq).issubset(valid)


# ------------------------------------------------------------------
# Augmentation pipelines
# ------------------------------------------------------------------

def get_train_transforms(img_size):
    h, w = img_size
    return A.Compose([
        A.RandomResizedCrop(size=(h, w), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    h, w = img_size
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class IDDSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, label_map,
                 transforms=None, ignore_index=255):
        self.img_dir      = img_dir
        self.mask_dir     = mask_dir
        self.label_map    = label_map
        self.transforms   = transforms
        self.ignore_index = ignore_index
        self.num_classes  = config.NUM_CLASSES

        exts = {'.jpg', '.jpeg', '.png'}
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        assert len(self.img_names) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name  = self.img_names[idx]
        base_name = os.path.splitext(img_name)[0]

        img  = np.array(
            Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        )
        mask = np.array(
            Image.open(
                os.path.join(self.mask_dir, base_name + '.png')
            ).convert('L')
        )

        # Label2/Label3 datasets are often already encoded as contiguous
        # train ids. Auto mode avoids collapsing valid classes by remapping
        # those masks as if they were raw IDD ids.
        if MASK_ENCODING_MODE == 'train_id':
            pass
        elif MASK_ENCODING_MODE == 'raw_id':
            mask = remap_mask(mask, self.label_map, self.ignore_index)
        else:  # auto
            if not _mask_already_train_ids(mask, self.num_classes,
                                           self.ignore_index):
                mask = remap_mask(mask, self.label_map, self.ignore_index)

        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img  = aug['image']
            mask = aug['mask'].long()

        return img, mask


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def get_loaders(label_level=None, batch_size=None, num_workers=None,
                img_size=None, ignore_index=None):
    label_level  = label_level  or config.LABEL_LEVEL
    batch_size   = batch_size   or config.BATCH_SIZE
    num_workers  = num_workers  if num_workers is not None else config.NUM_WORKERS
    img_size     = img_size     or config.IMG_SIZE
    ignore_index = ignore_index or config.IGNORE_INDEX

    label_map = LABEL_MAPS[label_level]

    train_ds = IDDSegDataset(
        config.TRAIN_IMG, config.TRAIN_MASK, label_map,
        get_train_transforms(img_size), ignore_index,
    )
    val_ds = IDDSegDataset(
        config.VAL_IMG, config.VAL_MASK, label_map,
        get_val_transforms(img_size), ignore_index,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Mask encoding mode: {MASK_ENCODING_MODE}")
    return train_loader, val_loader

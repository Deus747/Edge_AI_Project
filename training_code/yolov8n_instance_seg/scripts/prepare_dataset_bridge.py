import os
import shutil
from pathlib import Path

import yaml

import config


def _ensure_exists(path: str, kind: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {kind}: {path}")


def _relink(dst: str, src: str):
    p = Path(dst)
    if p.exists() or p.is_symlink():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(dst)
        else:
            p.unlink()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)


def _reset_dir(path: str):
    p = Path(path)
    if p.exists() or p.is_symlink():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(path)
        else:
            p.unlink()
    os.makedirs(path, exist_ok=True)


def _link_image_files(src_dir: str, dst_dir: str):
    """
    Link image files into bridge/images/* so Ultralytics resolves labels
    against bridge/labels/* instead of original IDD train/labels dirs.
    """
    _reset_dir(dst_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for f in Path(src_dir).iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts:
            continue
        dst = Path(dst_dir) / f.name
        try:
            # Hardlink preferred (fast, keeps bridge path semantics).
            os.link(str(f), str(dst))
        except OSError:
            # Fallback to symlink if hardlink unavailable.
            os.symlink(str(f), str(dst))


def main():
    _ensure_exists(config.TRAIN_IMG_SRC, "train image dir")
    _ensure_exists(config.VAL_IMG_SRC, "val image dir")
    _ensure_exists(config.TRAIN_LABEL_SRC, "train label dir")
    _ensure_exists(config.VAL_LABEL_SRC, "val label dir")

    _link_image_files(config.TRAIN_IMG_SRC, config.BRIDGE_IMG_TRAIN)
    _link_image_files(config.VAL_IMG_SRC, config.BRIDGE_IMG_VAL)
    _relink(config.BRIDGE_LBL_TRAIN, config.TRAIN_LABEL_SRC)
    _relink(config.BRIDGE_LBL_VAL, config.VAL_LABEL_SRC)

    # Basic label sanity: ensure class ids fit the declared names mapping.
    max_cls = -1
    for split_dir in (config.TRAIN_LABEL_SRC, config.VAL_LABEL_SRC):
        for txt in Path(split_dir).glob("*.txt"):
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id = int(line.split()[0])
                    if cls_id > max_cls:
                        max_cls = cls_id
    if max_cls >= len(config.NAMES):
        raise ValueError(
            f"Label class id {max_cls} exceeds configured names size {len(config.NAMES)}"
        )

    data = {
        "path": config.BRIDGE_ROOT,
        "train": "images/train",
        "val": "images/val",
        "names": config.NAMES,
    }
    with open(config.DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("Prepared YOLO dataset bridge")
    print(f"  Label level : {config.LABEL_LEVEL}")
    print(f"  YAML        : {config.DATA_YAML}")
    print(f"  Images src  : {config.TRAIN_IMG_SRC} / {config.VAL_IMG_SRC}")
    print(f"  Labels src  : {config.TRAIN_LABEL_SRC} / {config.VAL_LABEL_SRC}")


if __name__ == "__main__":
    main()

# Training Code

This folder contains only the reusable training source required for the baseline and logit-KD experiments, plus YOLOv8n-seg instance training. Use the top-level `dataset_preprocessing/` folder before training if you still need to convert raw IDD polygon annotations into masks.

Install dependencies from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Semantic Projects

Each semantic subfolder is self-contained and includes its own `requirements.txt`.

| Folder | Contents |
| --- | --- |
| `original_mbv4_teacher_student/` | MobileNetV4-L DeepLabV3+ teacher, MobileNetV4-S LR-ASPP baseline, and logit KD. |
| `convnext_upernet_to_mbv4small_lraspp/` | ConvNeXt-Base UPerNet teacher and MobileNetV4-S LR-ASPP logit KD. |
| `convnext_to_mbv4large_deeplabv3p/` | MobileNetV4-L DeepLabV3+ student logit KD from ConvNeXt teacher. |
| `convnext_to_mbv4small_deeplabv3p/` | MobileNetV4-S DeepLabV3+ student logit KD from ConvNeXt teacher. |

Typical semantic usage:

```bash
cd training_code/original_mbv4_teacher_student
pip install -r requirements.txt
python train_teacher.py
python train_baseline.py
python train_logit_kd.py
```

For projects that do not include `train_baseline.py`, run the teacher or provide the expected teacher checkpoint, then run:

```bash
python train_logit_kd.py
```

Use `IDD_LABEL_LEVEL` to select the semantic label level where supported:

```bash
IDD_LABEL_LEVEL=Label2ID python train_logit_kd.py
```

On PowerShell:

```powershell
$env:IDD_LABEL_LEVEL="Label2ID"
python train_logit_kd.py
```

## Logit KD Objective

The logit-KD objective is:

```text
loss = CE(student_logits, labels) + alpha * KL(teacher_logits / T, student_logits / T)
```

The project used temperature `T=4.0` and `alpha=1.0` in the semantic KD experiments.

## YOLOv8n-Seg Training

The YOLO code is in `yolov8n_instance_seg/`.

```bash
cd training_code/yolov8n_instance_seg
pip install -r requirements.txt
python scripts/prepare_dataset_bridge.py
python train_yolov8n_seg.py
```

Update `config.py` before running so it points to your local IDD image and instance-label folders.

# Real-Time Semantic and Instance Segmentation for Real-World Driving Environments

![IDD Edge AI Demo Input](docs/images/idd_sample_input.png)
![IDD Edge AI Demo Output](docs/images/idd_sample_label_mask.png)

An end-to-end **Edge AI road-scene perception pipeline** built on the **India Driving Dataset (IDD)**, combining dense semantic segmentation with **YOLOv8n-seg instance segmentation** for dynamic road users — trained, distilled, and deployed on a **Raspberry Pi with a Hailo accelerator**.

---

## Highlights

- **End-to-end edge AI pipeline** — Raw IDD polygon annotations are converted into masks, models are trained, compiled for Hailo, and deployed on Raspberry Pi.
- **Hybrid perception system** — Semantic segmentation handles dense road layout understanding, while YOLOv8n-seg provides object-level masks for dynamic road users.
- **Multiple semantic deployment models** — Three Hailo semantic models are available for different speed/accuracy trade-offs.
- **Logit knowledge distillation** — Compact student models are trained using softened teacher predictions to improve edge-device performance.
- **Hailo-ready deployment** — Final `.hef` weights are included for the Raspberry Pi demo.
- **Clean repository structure** — Logs, scheduler files, calibration data, ONNX/HAR intermediates, and training checkpoints are intentionally excluded.

---

## Repository Structure

```text
.
├── dataset_preprocessing/      # IDD polygon JSON → semantic/instance mask conversion
├── training_code/              # Semantic baseline, logit KD, and YOLOv8n-seg training
├── hailo_compilation/          # ONNX export and Hailo compile scripts
├── rpi_deployment/             # Raspberry Pi demo script and HEF weights
├── docs/images/                # README images and visual examples
└── README.md                   # Project overview and documentation
```

Each top-level folder contains its own `requirements.txt`.

---

## Motivation

Autonomous and assisted driving systems must reliably interpret road scenes under challenging real-world conditions. Indian traffic environments present a particularly demanding setting due to dense, unstructured layouts; wide variation in vehicle types and object scales; frequent occlusions and mixed traffic; and complex scene categories spanning road surfaces, sidewalks, barriers, vegetation, construction zones, riders, two-wheelers, and large commercial vehicles.

Cloud-based perception is incompatible with the latency requirements of embedded systems. This project develops a perception stack that runs entirely on edge hardware, delivering real-time semantic and instance-level scene understanding without relying on remote compute.

---

## Project Objectives

The system is designed to:

- Train semantic segmentation models across multiple label granularities
- Apply logit knowledge distillation to produce compact, high-quality student models
- Train a YOLOv8n-seg model for dynamic foreground object instance segmentation
- Convert all trained models into Hailo-compatible HEF files
- Run a live Raspberry Pi demo with semantic overlay, YOLO overlay, and on-the-fly model switching

The final pipeline demonstrates:

> **Dataset preprocessing → semantic & YOLO training → logit knowledge distillation → Hailo compilation → Raspberry Pi deployment**

---

## Hardware and Software

### Hardware

- Raspberry Pi
- Hailo AI accelerator
- Camera or video input source
- Development machine or WSL2 environment (for Hailo compilation)

### Software and Tools

- Python, PyTorch, TorchVision, TIMM
- Ultralytics YOLO, OpenCV, PyQt5
- Hailo Dataflow Compiler, HailoRT
- Jupyter Notebook

---

## Dataset Preprocessing

**Dataset link:** https://idd.insaan.iiit.ac.in/dataset/details/

The preprocessing notebook `dataset_preprocessing/idd_polygon_to_masks_preprocessing.ipynb` converts raw IDD polygon JSON annotations into dense pixel masks. The main entry point is:

```python
run_pipeline(
    datadir,
    out_basedir,
    encoding,
    do_semantic,
    do_instance,
    do_color,
    do_panoptic,
)
```

Supported label encodings:

| Encoding | Description |
|----------|-------------|
| `level1Id` | 7-class coarse semantic labels |
| `level2Id` | 16-class labels used for deployment |
| `level3Id` | 26-class fine semantic labels |
| `id`, `csId`, `csTrainId`, `level4Id`, `unifiedId` | Alternate encodings |

Expected directory layout after conversion:

```text
IDDL2/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Semantic masks use class IDs from `0` to `C−1`. The value `255` is reserved as the ignore label.

---

## Semantic Segmentation Models

Four architectures are evaluated, covering a range from lightweight deployment models to a powerful teacher network used for distillation.

| Model | Architecture | Role |
|-------|-------------|------|
| Model 1 | MobileNetV4-S + LR-ASPP | Smallest semantic deployment model |
| Model 2 | MobileNetV4-L + DeepLabV3+ | Higher-accuracy semantic deployment model |
| Model 3 | MobileNetV4-S + DeepLabV3+ | Balanced semantic deployment model |
| Teacher | ConvNeXt-Base + UPerNet | Strong teacher for knowledge distillation |

The semantic branch predicts **16 Label2ID classes** for the deployed system.

![Semantic validation prediction](docs/images/semantic_label2_prediction.png)

---

## Logit Knowledge Distillation

Student models are trained using a combination of ground-truth supervision and softened teacher predictions. The training objective is:

```
loss = CE(student_logits, labels) + α · KL(teacher_logits / T, student_logits / T)
```

where:

- `CE` — supervised cross-entropy loss
- `KL` — Kullback–Leibler divergence between teacher and student soft distributions
- `T` — distillation temperature
- `α` — distillation weight

The main experiments used `T = 4.0` and `α = 1.0`.

![Semantic logit-KD training curve](docs/images/semantic_label2_logit_kd_curve.png)

---

## YOLOv8n-Seg Instance Segmentation

YOLOv8n-seg is used to segment dynamic foreground objects, complementing the dense semantic branch with object-level masks for road users and vehicles.

**YOLO classes:**

| ID | Class |
|----|-------|
| 0 | person_animal |
| 1 | rider |
| 2 | motorcycle_bicycle |
| 3 | autorickshaw_car |
| 4 | large_vehicle |

**Best results from training logs:**

| Metric | Value |
|--------|------:|
| Best box mAP50-95 | 0.210 |
| Best mask mAP50-95 | 0.156 |
| Best mask mAP50 | 0.326 |

![YOLO validation prediction](docs/images/yolo_validation_prediction.jpg)

![YOLO training results](docs/images/yolo_training_results.png)

---

## Training Results

**Semantic segmentation:**

| Deployment Model | Architecture | Label2 mIoU |
|-----------------|-------------|------------:|
| Model 1 | MobileNetV4-S + LR-ASPP | 59.67 (before YOLO-aware fine-tune) |
| Model 2 | MobileNetV4-L + DeepLabV3+ | 68.15 (with logit KD) |
| Model 3 | MobileNetV4-S + DeepLabV3+ | 63.99 (with logit KD) |

**HEF deployment artifacts included in this repository:**

| File | Model | Size | FPS on RPi |
|------|-------|-----:|-----------|
| `rpi_deployment/weights/model1_hailo.hef` | Model 1 semantic | 1.34 MiB | — |
| `rpi_deployment/weights/model2_hailo.hef` | Model 2 semantic | 6.60 MiB | — |
| `rpi_deployment/weights/model3_hailo.hef` | Model 3 semantic | 9.93 MiB | — |
| `rpi_deployment/weights/yolov8n_seg_hailo.hef` | YOLOv8n-seg | 7.07 MiB | — |

> Fill in the **FPS on RPi** column with measured values from your Raspberry Pi benchmarks.

---

## Hailo Compilation

The compilation code is in `hailo_compilation/`. The full pipeline is:

```
PyTorch checkpoint (.pth / .pt)
  → ONNX
  → HAR
  → Optimized HAR
  → HEF
```

- **ONNX** — portable inference graph
- **HAR** — Hailo Archive used by the Hailo Dataflow Compiler
- **HEF** — Hailo Executable Format loaded by HailoRT on Raspberry Pi

Generated ONNX, HAR, optimized HAR, compiled HAR, calibration tensors, and compiler logs are not included. Place your own checkpoint in the relevant model folder before running the export and compile scripts.

```bash
cd hailo_compilation/semantic_model3_mbv4small_deeplabv3p
python export_onnx.py
python prep_calib.py --images /path/to/calibration/images --out calib_set.npy
bash hailo_compile.sh
```

---

## Raspberry Pi Deployment

The deployment code is in `rpi_deployment/`.

```bash
cd rpi_deployment
pip install -r requirements.txt
python python_if_models_switch_pipelined.py
```

**Benchmark mode:**

```bash
python python_if_models_switch_pipelined.py --benchmark /path/to/video.mp4 300
```

The application supports video input, camera input, semantic model switching, YOLO overlay toggling, overlay alpha control, and Hailo HEF inference.

---

## Quick Start

### 1. Preprocess IDD

```bash
cd dataset_preprocessing
pip install -r requirements.txt
jupyter notebook idd_polygon_to_masks_preprocessing.ipynb
```

### 2. Train semantic models

```bash
cd training_code/original_mbv4_teacher_student
pip install -r requirements.txt
python train_teacher.py
python train_baseline.py
python train_logit_kd.py
```

For other student folders:

```bash
python train_logit_kd.py
```

### 3. Train YOLO

```bash
cd training_code/yolov8n_instance_seg
pip install -r requirements.txt
python scripts/prepare_dataset_bridge.py
python train_yolov8n_seg.py
```

### 4. Compile for Hailo

Use Ubuntu under WSL2 or a native Linux environment with the Hailo Dataflow Compiler installed.

```bash
cd hailo_compilation
pip install -r requirements.txt
```

Then enter the target model folder and run its export and compile scripts.

### 5. Run on Raspberry Pi

```bash
cd rpi_deployment
pip install -r requirements.txt
python python_if_models_switch_pipelined.py
```

### 6. Example 


<img width="300" height="377" alt="image" src="https://github.com/user-attachments/assets/2ec9d19c-1697-4e84-be0e-7c3140da5f81" />

#### Running real time on a RPI 
---

## Planned Improvements

- Larger and more representative calibration sets for improved Hailo quantization
- Improved mask quality for using temporal smoothing
- Additional semantic classes and panoptic fusion experiments
- Introduce REID for better tracking of instance objects

---

## Team

| Name | Role |
|------|------|
| Debanshu Mallick | Project member |
| Chandan Rai | Project member |
| Tamaghna Mandal | Project member |
| Yuvaraj DC | Project member |

**Mentor / Supervisor:** Pandarasamy Arjunan — RBCCPS, Indian Institute of Science

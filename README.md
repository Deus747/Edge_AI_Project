# Real Time Semantic and Instance Segmentation for Real-World Driving Environments

![IDD Edge AI Demo Input](docs/images/idd_sample_input.png)

This project implements an end-to-end **Edge AI road-scene perception pipeline** using the **India Driving Dataset (IDD)**. The system combines **semantic segmentation** for dense scene understanding with **YOLOv8n-seg instance segmentation** for dynamic road users, then deploys the final models on a **Raspberry Pi with a Hailo accelerator**.

The repository contains the cleaned code needed for dataset preprocessing, semantic model training, logit knowledge distillation, YOLO training, Hailo compilation, and Raspberry Pi deployment.

## Highlights

* **End-to-end Edge AI pipeline**: Raw IDD polygon annotations are converted into masks, models are trained, compiled for Hailo, and deployed on Raspberry Pi.
* **Hybrid perception system**: Semantic segmentation handles dense road layout, while YOLOv8n-seg handles dynamic object instances.
* **Multiple semantic deployment models**: Three Hailo semantic models are provided for different speed/accuracy trade-offs.
* **Logit knowledge distillation**: Compact student models are trained using teacher logits to improve edge-device performance.
* **Hailo-ready deployment**: Final `.hef` weights are included for the Raspberry Pi demo.
* **Clean GitHub structure**: Logs, scheduler files, calibration data, ONNX/HAR intermediates, and training checkpoints are intentionally excluded.

---

## Repository Structure

```text
.
├── dataset_preprocessing/      # IDD polygon JSON to semantic/instance mask conversion
├── training_code/              # Semantic baseline, logit KD, and YOLOv8n-seg training
├── hailo_compilation/          # ONNX export and Hailo compile scripts
├── rpi_deployment/             # Raspberry Pi demo script and HEF weights
├── docs/images/                # README images and visual examples
└── README.md                   # Project overview and documentation
```

Each top-level folder has its own `requirements.txt`.

---

## Problem Statement

Autonomous and assisted-driving systems require reliable understanding of road scenes. This is especially challenging in Indian traffic conditions because of:

* Dense and unstructured road environments
* Large variation in vehicle types and object scales
* Occlusions, mixed traffic, and non-lane-based driving
* Complex scene categories such as road, sidewalk, barriers, vegetation, construction, riders, two-wheelers, and large vehicles

Cloud-based perception is unsuitable for low-latency embedded systems. The goal of this project is to build a perception stack that can run locally on edge hardware and provide real-time semantic and instance-level understanding of IDD road scenes.

---

## Project Objectives

The main objective is to develop a deployable edge perception system that:

* Converts IDD polygon annotations into trainable segmentation masks
* Trains semantic segmentation models at multiple label granularities
* Uses logit knowledge distillation to train compact student models
* Trains a YOLOv8n-seg model for foreground object instances
* Converts trained models into Hailo-compatible HEF files
* Runs a Raspberry Pi demo with semantic overlay, YOLO overlay, and model switching

The final system demonstrates:

> IDD dataset preprocessing -> semantic and YOLO training -> logit KD -> Hailo compilation -> Raspberry Pi deployment.

---

## Hardware and Software Used

### Hardware

* Raspberry Pi
* Hailo AI accelerator
* Camera or video input source
* Development machine or WSL environment for Hailo compilation

### Software and Tools

* Python
* PyTorch
* TorchVision
* TIMM
* Ultralytics YOLO
* OpenCV
* PyQt5
* Hailo Dataflow Compiler
* HailoRT
* Jupyter Notebook

---

## Dataset Preprocessing

The dataset preprocessing code is in:

```text
dataset_preprocessing/
```

The notebook:

```text
idd_polygon_to_masks_preprocessing.ipynb
```

converts raw IDD polygon JSON annotations into dense masks.

The main conversion function is:

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

Supported label encodings include:

* `level1Id`: 7-class coarse semantic labels
* `level2Id`: 16-class semantic labels used for deployment
* `level3Id`: 26-class fine semantic labels
* `id`, `csId`, `csTrainId`, `level4Id`, `unifiedId`: alternate encodings

Expected training layout after conversion:

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

Semantic masks use valid class IDs from `0` to `C-1`, with `255` used as the ignore label.

---

## Semantic Segmentation Models

The project evaluates multiple semantic segmentation architectures for IDD.

| Model | Architecture | Role |
| --- | --- | --- |
| Model 1 | MobileNetV4-S + LR-ASPP | Smallest semantic deployment model |
| Model 2 | MobileNetV4-L + DeepLabV3+ | Higher-accuracy semantic deployment model |
| Model 3 | MobileNetV4-S + DeepLabV3+ | Balanced semantic deployment model |
| Teacher | ConvNeXt-Base + UPerNet | Strong teacher for distillation |

The semantic branch predicts **16 Label2ID classes** for the deployed system.

![Semantic validation prediction](docs/images/semantic_label2_prediction.png)

---

## Logit Knowledge Distillation

Only **logit KD** is included in this cleaned repository. The student model is trained using ground-truth labels and softened teacher predictions.

The loss is:

```text
loss = CE(student_logits, labels) + alpha * KL(teacher_logits / T, student_logits / T)
```

where:

* `CE` is supervised cross entropy
* `KL` is Kullback-Leibler divergence between teacher and student logits
* `T` is the distillation temperature
* `alpha` controls the distillation strength

The main experiments used:

```text
T = 4.0
alpha = 1.0
```

![Semantic logit-KD curve](docs/images/semantic_label2_logit_kd_curve.png)

---

## YOLOv8n-Seg Instance Segmentation

YOLOv8n-seg is used to segment dynamic foreground objects. This complements semantic segmentation by giving object-level masks for road users and vehicles.

YOLO classes:

| ID | Class |
| --- | --- |
| 0 | person_animal |
| 1 | rider |
| 2 | motorcycle_bicycle |
| 3 | autorickshaw_car |
| 4 | large_vehicle |

Best YOLO result from the inspected training logs:

| Metric | Value |
| --- | ---: |
| Best box mAP50-95 | 0.21013 |
| Best mask mAP50-95 | 0.15582 |
| Best mask mAP50 | 0.32560 |

![YOLO validation prediction](docs/images/yolo_validation_prediction.jpg)

![YOLO training results](docs/images/yolo_training_results.png)

---

## Training Results

Representative semantic results:

| Deployment Model | Architecture | Main Result |
| --- | --- | ---: |
| Model 1 | MobileNetV4-S + LR-ASPP | 59.67 Label2 mIoU before YOLO-aware fine-tune |
| Model 2 | MobileNetV4-L + DeepLabV3+ | 68.15 Label2 mIoU with logit KD |
| Model 3 | MobileNetV4-S + DeepLabV3+ | 63.99 Label2 mIoU with logit KD |

HEF deployment artifacts included in this repository:

| File | Model | Size |
| --- | --- | ---: |
| `rpi_deployment/weights/model1_hailo.hef` | Model 1 semantic | 1.34 MiB |
| `rpi_deployment/weights/model2_hailo.hef` | Model 2 semantic | 6.60 MiB |
| `rpi_deployment/weights/model3_hailo.hef` | Model 3 semantic | 9.93 MiB |
| `rpi_deployment/weights/yolov8n_seg_hailo.hef` | YOLOv8n-seg | 7.07 MiB |

---

## Hailo Compilation

The Hailo compilation code is in:

```text
hailo_compilation/
```

The compilation flow is:

```text
PyTorch checkpoint (.pth or .pt)
  -> ONNX
  -> HAR
  -> optimized HAR
  -> HEF
```

ONNX is the portable inference graph. HAR is the Hailo Archive used by the Hailo compiler. HEF is the final Hailo Executable Format loaded by HailoRT on Raspberry Pi.

Generated ONNX, HAR, optimized HAR, compiled HAR, calibration tensors, and compiler logs are not included. Place your own checkpoint in the relevant model folder before running export and compile scripts.

Example:

```bash
cd hailo_compilation/semantic_model3_mbv4small_deeplabv3p
python export_onnx.py
python prep_calib.py --images /path/to/calibration/images --out calib_set.npy
bash hailo_compile.sh
```

---

## Raspberry Pi Deployment

The deployment code is in:

```text
rpi_deployment/
```

Run the demo:

```bash
cd rpi_deployment
pip install -r requirements.txt
python python_if_models_switch_pipelined.py
```

Benchmark mode:

```bash
python python_if_models_switch_pipelined.py --benchmark /path/to/video.mp4 300
```

The application supports:

* Video input
* Camera input
* Semantic model switching
* YOLO overlay toggle
* Overlay alpha control
* Hailo HEF inference

---

## Quick Start

### 1. Preprocess IDD

```bash
cd dataset_preprocessing
pip install -r requirements.txt
jupyter notebook idd_polygon_to_masks_preprocessing.ipynb
```

### 2. Train Semantic Models

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

Use Ubuntu under WSL2 or a Linux environment with Hailo Dataflow Compiler installed.

```bash
cd hailo_compilation
pip install -r requirements.txt
```

Then enter the target model folder and run its export/compile scripts.

### 5. Run on Raspberry Pi

```bash
cd rpi_deployment
pip install -r requirements.txt
python python_if_models_switch_pipelined.py
```

---

## Planned Improvements

Future improvements can include:

* Larger and more representative calibration sets for Hailo quantization
* Additional real-world Raspberry Pi benchmarking
* Improved YOLO instance masks for smaller road users
* Additional semantic classes or panoptic fusion
* A cleaner real-time dashboard for deployment visualization
* Model pruning or architecture search for faster edge inference

---

## Team

Project members:

* **[Name 1]**
* **[Name 2]**
* **[Name 3]**

Mentor / Supervisor:

* **[Mentor Name]**, **[Department / Lab / Institution]**

Contact:

* **[email@example.com]**


# IDD Edge AI Segmentation on Raspberry Pi + Hailo

This repository contains the cleaned code needed to reproduce and deploy an edge perception pipeline for the India Driving Dataset (IDD). The system combines semantic segmentation for dense road-scene layout with YOLOv8n-seg instance segmentation for dynamic objects, then deploys both on Raspberry Pi with a Hailo accelerator.

The repository is intentionally source-focused. It includes reusable training, preprocessing, Hailo compilation, and Raspberry Pi deployment code. It excludes training logs, scheduler files, calibration datasets, generated ONNX/HAR files, compiler logs, and training checkpoints.

![IDD sample input](docs/images/idd_sample_input.png)

## Pipeline

```text
Raw IDD polygons
  -> dataset_preprocessing/
  -> semantic masks and instance masks
  -> training_code/
  -> semantic teacher/student + logit KD and YOLOv8n-seg
  -> hailo_compilation/
  -> ONNX -> HAR -> HEF
  -> rpi_deployment/
  -> Raspberry Pi real-time demo
```

## Repository Folders

Each top-level folder has its own `requirements.txt`.

| Folder | What it contains |
| --- | --- |
| `dataset_preprocessing/` | Notebook to convert raw IDD polygon JSON annotations into semantic and instance masks. |
| `training_code/` | Semantic baseline/logit-KD training code and YOLOv8n-seg training code. |
| `hailo_compilation/` | Minimal ONNX export and Hailo compile scripts for the semantic and YOLO models. |
| `rpi_deployment/` | PyQt/OpenCV Raspberry Pi demo and deployed Hailo `.hef` weights. |

## Model Summary

The semantic branch predicts 16 Label2ID classes for deployment. YOLO handles foreground dynamic object instances.

| Deployment model | Architecture | Role | Main result |
| --- | --- | --- | --- |
| Model 1 | MobileNetV4-S + LR-ASPP | Smallest semantic model | 59.67 Label2 mIoU before YOLO-aware fine-tune |
| Model 2 | MobileNetV4-L + DeepLabV3+ | Highest semantic accuracy | 68.15 Label2 mIoU with logit KD |
| Model 3 | MobileNetV4-S + DeepLabV3+ | Balanced semantic model | 63.99 Label2 mIoU with logit KD |
| YOLO | YOLOv8n-seg | Dynamic-object instance masks | 0.15582 best mask mAP50-95 |

![Semantic prediction](docs/images/semantic_label2_prediction.png)

## Logit KD

Only logit KD is included in this cleaned repository. The student is trained with supervised cross entropy and teacher-logit KL divergence:

```text
loss = CE(student_logits, labels) + alpha * KL(teacher_logits / T, student_logits / T)
```

The project used `T=4.0` and `alpha=1.0` for the main KD runs. Extra KD variants from the original experiments were removed from this upload.

![Semantic logit-KD curve](docs/images/semantic_label2_logit_kd_curve.png)

## YOLO Instance Segmentation

The YOLO branch uses five grouped IDD instance classes:

| ID | Class |
| --- | --- |
| 0 | person_animal |
| 1 | rider |
| 2 | motorcycle_bicycle |
| 3 | autorickshaw_car |
| 4 | large_vehicle |

![YOLO validation prediction](docs/images/yolo_validation_prediction.jpg)

![YOLO training results](docs/images/yolo_training_results.png)

## Hailo Compilation

The Hailo compilation code follows this flow:

```text
PyTorch checkpoint (.pth or .pt)
  -> ONNX export
  -> Hailo Archive (.har)
  -> optimized HAR
  -> Hailo Executable Format (.hef)
```

ONNX is the portable inference graph. HAR is Hailo's parsed and optimized graph container. HEF is the final binary loaded by HailoRT on Raspberry Pi.

Generated ONNX/HAR/HEF files are not included in `hailo_compilation/`. Place your trained checkpoint into the relevant model folder, generate calibration data from representative IDD images, then run that folder's export and compile scripts.

| Model | HEF included for RPi | HEF size |
| --- | --- | ---: |
| Model 1 semantic | `rpi_deployment/weights/model1_hailo.hef` | 1.34 MiB |
| Model 2 semantic | `rpi_deployment/weights/model2_hailo.hef` | 6.60 MiB |
| Model 3 semantic | `rpi_deployment/weights/model3_hailo.hef` | 9.93 MiB |
| YOLOv8n-seg | `rpi_deployment/weights/yolov8n_seg_hailo.hef` | 7.07 MiB |

## Quick Start

### 1. Preprocess IDD

```bash
cd dataset_preprocessing
pip install -r requirements.txt
jupyter notebook idd_polygon_to_masks_preprocessing.ipynb
```

Update the dataset paths inside the notebook, then generate semantic and/or instance masks.

### 2. Train Models

```bash
cd training_code/original_mbv4_teacher_student
pip install -r requirements.txt
python train_teacher.py
python train_baseline.py
python train_logit_kd.py
```

For folders without a baseline script, provide or train the expected teacher checkpoint, then run:

```bash
python train_logit_kd.py
```

YOLO training:

```bash
cd training_code/yolov8n_instance_seg
pip install -r requirements.txt
python scripts/prepare_dataset_bridge.py
python train_yolov8n_seg.py
```

### 3. Compile for Hailo in WSL

Use Ubuntu under WSL2 for the Hailo Dataflow Compiler.

```bash
cd hailo_compilation
pip install -r requirements.txt
```

Install the Hailo Dataflow Compiler wheel from the Hailo Developer Zone, then enter the target model folder:

```bash
cd semantic_model3_mbv4small_deeplabv3p
python export_onnx.py
python prep_calib.py --images /path/to/calibration/images --out calib_set.npy
bash hailo_compile.sh
```

The Hailo wheel and calibration data are not included in this repository.

### 4. Run on Raspberry Pi

Install Raspberry Pi dependencies and HailoRT, then run:

```bash
cd rpi_deployment
pip install -r requirements.txt
python python_if_models_switch_pipelined.py
```

Benchmark mode:

```bash
python python_if_models_switch_pipelined.py --benchmark /path/to/video.mp4 300
```

The GUI supports video input, camera input, semantic model switching, YOLO overlay toggling, and overlay alpha control.

## What Is Not Included

- Raw IDD dataset files.
- Training checkpoints (`.pth`, `.pt`) for retraining or compilation.
- Generated ONNX, HAR, optimized HAR, compiled HAR, and compiler logs.
- Calibration images and calibration tensors.
- Training logs and generated plot archives.

The included `rpi_deployment/weights/` folder contains the final HEF files needed to run the Raspberry Pi demo.


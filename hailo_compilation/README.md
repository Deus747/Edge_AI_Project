# Hailo Compilation

This folder contains the minimal code needed to export trained checkpoints to ONNX and compile them for Hailo. Generated files from previous Hailo runs are intentionally excluded.

Install the Python-side export dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Folder Contents

| Folder | Model |
| --- | --- |
| `semantic_model1_mbv4small_lraspp/` | MobileNetV4-S LR-ASPP semantic model. |
| `semantic_model2_mbv4large_deeplabv3p/` | MobileNetV4-L DeepLabV3+ semantic model. |
| `semantic_model3_mbv4small_deeplabv3p/` | MobileNetV4-S DeepLabV3+ semantic model. |
| `yolov8n_seg/` | YOLOv8n-seg instance model. |

Each model folder contains:

- `export_onnx.py`: converts a PyTorch checkpoint to ONNX.
- `*.alls`: Hailo model script with normalization and compile directives.
- `hailo_compile.sh`: parse, optimize, and compile commands.
- `prep_calib.py`: helper for creating calibration tensors from images.
- `model.py`: included for semantic models that need local architecture definitions.

## Hailo Setup in WSL

Hailo Dataflow Compiler is distributed as a Linux Python package. On Windows, use Ubuntu under WSL2.

1. Install WSL2 from an administrator PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

2. Open Ubuntu and update packages:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-dev python3-pip python3-venv python3-tk graphviz graphviz-dev build-essential virtualenv unzip
```

3. Download the matching Hailo Dataflow Compiler wheel from the Hailo Developer Zone. Use the version that matches your Hailo software suite.

4. Create a virtual environment and install the wheel:

```bash
python3 -m venv hailo_dfc_env
source hailo_dfc_env/bin/activate
pip install --upgrade pip
pip install /path/to/hailo_dataflow_compiler-*.whl
pip install -r requirements.txt
```

5. Verify that the compiler package imports:

```bash
python -c "from hailo_sdk_client import ClientRunner; print('Hailo DFC import OK')"
```

Notes:

- WSL compilation can work without GPU acceleration, but advanced optimization may be limited.
- Keep Hailo Dataflow Compiler, HailoRT, and target device software versions compatible.
- The Hailo wheel is not included in this repository.

## Compile Flow

For each semantic model:

```bash
cd hailo_compilation/semantic_model1_mbv4small_lraspp
# place model1.pth here
python export_onnx.py
python prep_calib.py --images /path/to/calibration/images --out calib_set.npy
bash hailo_compile.sh
```

For YOLO:

```bash
cd hailo_compilation/yolov8n_seg
# place best.pt here
python export_onnx.py
python prep_calib.py --images /path/to/calibration/images --out calib_set.npy
bash hailo_compile.sh
```

The expected generated flow is:

```text
PyTorch checkpoint -> ONNX -> parsed HAR -> optimized HAR -> HEF
```

Only source code is included here. The generated ONNX, HAR, optimized HAR, compiled HAR, HEF, logs, and calibration arrays are intentionally excluded.


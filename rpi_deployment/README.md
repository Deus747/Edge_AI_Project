# Raspberry Pi Deployment

This folder contains the Raspberry Pi demo code and the Hailo HEF weights required by the application.

Install dependencies on the Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

System packages commonly needed:

```bash
sudo apt update
sudo apt install -y python3-pyqt5 python3-opencv gstreamer1.0-tools v4l-utils
```

Install HailoRT and the Hailo Python bindings using the packages provided for your Raspberry Pi/Hailo kit from the Hailo Developer Zone or the official Raspberry Pi AI Kit documentation. The Python script requires:

```python
from hailo_platform import HEF, VDevice, InferVStreams
```

Verify Hailo runtime access:

```bash
hailortcli scan
python -c "import hailo_platform; print('Hailo runtime import OK')"
```

## Files

| Path | Purpose |
| --- | --- |
| `python_if_models_switch_pipelined.py` | PyQt/OpenCV demo application. |
| `weights/model1_hailo.hef` | MobileNetV4-S LR-ASPP semantic model. |
| `weights/model2_hailo.hef` | MobileNetV4-L DeepLabV3+ semantic model. |
| `weights/model3_hailo.hef` | MobileNetV4-S DeepLabV3+ semantic model. |
| `weights/yolov8n_seg_hailo.hef` | YOLOv8n-seg instance model. |

## Run the Demo

From this folder:

```bash
python python_if_models_switch_pipelined.py
```

The GUI supports video input, camera input, semantic model switching, YOLO overlay toggling, and overlay alpha control.

Benchmark mode:

```bash
python python_if_models_switch_pipelined.py --benchmark /path/to/video.mp4 300
```

If the script cannot find a camera, check `libcamera`, GStreamer, and `/dev/video*` availability on the Raspberry Pi.


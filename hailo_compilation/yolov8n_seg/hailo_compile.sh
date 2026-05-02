#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPORT_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
HAILO_BIN="${ROOT_DIR}/hailo_ai_sw_suite/hailo_venv/bin/hailo"
LOCAL_HOME="${SCRIPT_DIR}/.hailo_home"
YOLO_CONFIG_DIR="${SCRIPT_DIR}/.ultralytics"

if [[ ! -x "${EXPORT_PYTHON}" ]]; then
    echo "Missing local export python: ${EXPORT_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${HAILO_BIN}" ]]; then
    echo "Missing hailo CLI: ${HAILO_BIN}" >&2
    exit 1
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "${LOCAL_HOME}" "${YOLO_CONFIG_DIR}"
export HOME="${LOCAL_HOME}"
export YOLO_CONFIG_DIR

cd "${SCRIPT_DIR}"

if [[ ! -f yolov8n_seg_hailo.onnx ]]; then
    "${EXPORT_PYTHON}" export_onnx.py
fi

if [[ ! -d calibs_npy ]] || [[ -z "$(find calibs_npy -maxdepth 1 -type f -name '*.npy' -print -quit)" ]]; then
    "${EXPORT_PYTHON}" prep_calib.py
fi

"${HAILO_BIN}" parser onnx \
    yolov8n_seg_hailo.onnx \
    --hw-arch hailo8 \
    --end-node-names \
    /model.22/cv2.2/cv2.2.2/Conv \
    /model.22/cv3.2/cv3.2.2/Conv \
    /model.22/cv4.2/cv4.2.2/Conv \
    /model.22/cv2.1/cv2.1.2/Conv \
    /model.22/cv3.1/cv3.1.2/Conv \
    /model.22/cv4.1/cv4.1.2/Conv \
    /model.22/cv2.0/cv2.0.2/Conv \
    /model.22/cv3.0/cv3.0.2/Conv \
    /model.22/cv4.0/cv4.0.2/Conv \
    /model.22/proto/cv3/act/Mul \
    --har yolov8n_seg_hailo.har

"${HAILO_BIN}" optimize \
    yolov8n_seg_hailo.har \
    --hw-arch hailo8 \
    --calib-set-path ./calibs_npy \
    --model-script ./yolov8n_seg_hailo.alls \
    --output-har-path yolov8n_seg_hailo_optimized.har

"${HAILO_BIN}" compiler \
    yolov8n_seg_hailo_optimized.har \
    --model-script ./yolov8n_seg_hailo.alls \
    --hw-arch hailo8

echo
echo "Artifacts in ${SCRIPT_DIR}:"
ls -lh yolov8n_seg_hailo.onnx yolov8n_seg_hailo.har yolov8n_seg_hailo_optimized.har ./*.hef 2>/dev/null || true

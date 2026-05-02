#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/hailo_ai_sw_suite/hailo_venv/bin/python"
HAILO_BIN="${ROOT_DIR}/hailo_ai_sw_suite/hailo_venv/bin/hailo"
LOCAL_HOME="${SCRIPT_DIR}/.hailo_home"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Missing python interpreter: ${VENV_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${HAILO_BIN}" ]]; then
    echo "Missing hailo CLI: ${HAILO_BIN}" >&2
    exit 1
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "${LOCAL_HOME}"
export HOME="${LOCAL_HOME}"

cd "${SCRIPT_DIR}"

if [[ ! -f model1_hailo.onnx ]]; then
    "${VENV_PYTHON}" export_onnx.py
fi

if [[ ! -d calibs_npy ]] || [[ -z "$(find calibs_npy -maxdepth 1 -type f -name '*.npy' -print -quit)" ]]; then
    "${VENV_PYTHON}" prep_calib.py
fi

"${HAILO_BIN}" parser onnx \
    model1_hailo.onnx \
    --hw-arch hailo8 \
    --har model1.har

"${HAILO_BIN}" optimize \
    model1.har \
    --hw-arch hailo8 \
    --calib-set-path ./calibs_npy \
    --model-script ./model1.alls \
    --output-har-path model1_optimized.har

"${HAILO_BIN}" compiler \
    model1_optimized.har \
    --model-script ./model1.alls \
    --hw-arch hailo8

echo
echo "Artifacts in ${SCRIPT_DIR}:"
ls -lh model1_hailo.onnx model1.har model1_optimized.har ./*.hef 2>/dev/null || true

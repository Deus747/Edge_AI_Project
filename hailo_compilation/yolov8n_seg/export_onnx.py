#!/usr/bin/env python3

import pathlib

import onnx
from ultralytics import YOLO


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CHECKPOINT = SCRIPT_DIR / "best.pt"
OUTPUT_ONNX = SCRIPT_DIR / "yolov8n_seg_hailo.onnx"
IMGSZ = 512


def main():
    print("=" * 60)
    print("EXPORTING YOLOV8N SEGMENTATION FOR HAILO")
    print("=" * 60)

    print("\n1. Loading checkpoint...")
    model = YOLO(str(CHECKPOINT))
    print(f"   task: {model.task}")
    print(f"   classes: {len(model.names)}")
    print(f"   names: {model.names}")
    print(f"   imgsz: {model.overrides.get('imgsz')}")

    print("\n2. Exporting ONNX...")
    exported = pathlib.Path(
        model.export(
            format="onnx",
            imgsz=IMGSZ,
            opset=11,
            simplify=False,
            dynamic=False,
            nms=False,
            half=False,
            batch=1,
            device="cpu",
            verbose=False,
        )
    )
    if exported.resolve() != OUTPUT_ONNX.resolve():
        exported.replace(OUTPUT_ONNX)
    print(f"   wrote {OUTPUT_ONNX.name}")

    print("\n3. Verifying ONNX...")
    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)

    print(f"   inputs:  {[node.name for node in onnx_model.graph.input]}")
    print(f"   outputs: {[node.name for node in onnx_model.graph.output]}")

    input_shape = [
        d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    print(f"   input shape: {input_shape}")

    print("\n" + "=" * 60)
    print(f"SUCCESS: {OUTPUT_ONNX}")
    print("=" * 60)


if __name__ == "__main__":
    main()

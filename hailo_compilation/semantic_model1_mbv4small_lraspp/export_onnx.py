#!/usr/bin/env python3

import pathlib

import onnx
import torch

from model import CLASS_NAMES, build_model

try:
    from onnxsim import simplify
except ImportError:
    simplify = None


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CHECKPOINT = SCRIPT_DIR / "model1.pth"
OUTPUT_ONNX = SCRIPT_DIR / "model1_hailo.onnx"
NUM_CLASSES = 16
BB_CHANNELS = [32, 32, 64, 96]


def main():
    print("=" * 60)
    print("EXPORTING MODEL1 LR-ASPP MOBILENETV4-S FOR HAILO")
    print("=" * 60)

    print("\n1. Building model...")
    model = build_model(
        num_classes=NUM_CLASSES,
        bb_channels=BB_CHANNELS,
        is_export=True,
        device="cpu",
    )
    model.eval()
    print("   loaded model definition")
    print(f"   classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    print("\n2. Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = checkpoint["model_state"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch\nmissing={missing}\nunexpected={unexpected}"
        )
    print(f"   loaded weights from {CHECKPOINT.name}")
    print(f"   epoch: {checkpoint.get('epoch', 'n/a')}, mIoU: {checkpoint.get('miou', 'n/a')}")

    dummy_input = torch.randn(1, 3, 512, 512)

    print("\n3. Tracing with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    print("   tracing complete")

    print("\n4. Exporting ONNX...")
    torch.onnx.export(
        traced_model,
        dummy_input,
        str(OUTPUT_ONNX),
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"   wrote {OUTPUT_ONNX.name}")

    print("\n5. Verifying ONNX...")
    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)

    if simplify is not None:
        print("\n6. Simplifying ONNX...")
        model_simp, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("onnxsim reported a failed simplification check")
        onnx.save(model_simp, OUTPUT_ONNX)
        onnx_model = model_simp
        print("   simplification complete")
    else:
        print("\n6. Skipping simplification because onnxsim is not installed")

    input_shape = [
        d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    print(f"   input shape:  {input_shape}")
    print(f"   output shape: {output_shape}")

    print("\n" + "=" * 60)
    print(f"SUCCESS: {OUTPUT_ONNX}")
    print("=" * 60)


if __name__ == "__main__":
    main()

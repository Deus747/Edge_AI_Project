# IDD Edge AI Segmentation Project Report

This is a more detail report documents the complete project flow for semantic segmentation and instance segmentation on edge hardware: dataset post-processing, model training, logit knowledge distillation, quantization/compilation for Hailo, and Raspberry Pi deployment. GPU-cluster setup details are intentionally omitted, but training logs were inspected to recover the reported losses and validation metrics.
and contains all the detailed metrics
## 1. Project Scope

The project builds a real-time road-scene perception stack for the India Driving Dataset (IDD). The final deployed demo combines:

- A semantic segmentation model running as a Hailo HEF.
- A YOLOv8n-seg instance segmentation model running as a Hailo HEF.
- A Raspberry Pi PyQt/OpenCV application that overlays semantic classes and detected dynamic objects.
- Runtime switching between three semantic model variants.

## 2. End-to-End Pipeline

The full pipeline is:

1. Convert raw IDD polygon annotations into semantic masks, instance masks, color previews, or panoptic annotations.
2. Organize images and masks into train/validation/test splits for each label level.
3. Train strong semantic teachers on Label1ID, Label2ID, and Label3ID.
4. Train smaller semantic students using logit knowledge distillation.
5. Train YOLOv8n-seg on IDD instance classes using a dataset bridge.
6. Export selected PyTorch checkpoints to ONNX.
7. Parse ONNX to Hailo HAR, optimize/quantize with calibration images, and compile HEF files.
8. Deploy semantic and YOLO HEFs to Raspberry Pi with a PyQt/OpenCV/Hailo runtime.

## 3. IDD Dataset Post-Processing

The conversion logic is in `idd_polygon_to_mask.ipynb`. It reads the original IDD polygon annotations from `gtFine` and writes derived supervision targets.


The notebook exposes:

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

Supported encodings include:

| Encoding | Use |
| --- | --- |
| `level1Id` | Coarse 7-class semantic labels. |
| `level2Id` | Medium 16-class semantic labels. |
| `level3Id` | Fine 26-class semantic labels. |
| `id`, `csId`, `csTrainId`, `level4Id`, `unifiedId` | Alternate IDD/Cityscapes-compatible encodings. |

Generated outputs can include:

| Output | Description |
| --- | --- |
| Semantic PNG | Single-channel class-ID mask. |
| Instance PNG | Encodes instance IDs as `classId * 1000 + instanceIndex`. |
| Color PNG | Visual preview using the label color table. |
| Panoptic PNG/JSON | Panoptic-style category and segment records. |

Important conversion details:

- Unknown labels are skipped.
- Label names ending with `group` are normalized before lookup.
- Background/ignore pixels use ID `255` for semantic-style encodings.
- Instance counters are keyed by the actual encoded class value, which avoids collisions when using coarser label levels.
- Panoptic category IDs are generated from the selected encoding rather than being hardcoded to a single IDD level.

For semantic training, the project expects split folders like:

```text
<IDDLx>/train/images
<IDDLx>/train/masks
<IDDLx>/val/images
<IDDLx>/val/masks
<IDDLx>/test/images
<IDDLx>/test/masks
```

All semantic logs report:

| Split | Images |
| --- | ---: |
| Train | 12,872 |
| Validation | 1,995 |

## 4. Label Spaces

The semantic projects train on three IDD label levels.

### Label1ID

Label1ID is a 7-class coarse setup:

| ID | Class |
| ---: | --- |
| 0 | road |
| 1 | sky |
| 2 | vegetation |
| 3 | building |
| 4 | vehicle |
| 5 | person |
| 6 | background |

### Label2ID

Label2ID is a 16-class medium-granularity setup and is the label space used for all deployed models:

| ID | Class |
| ---: | --- |
| 0 | road |
| 1 | drivable-other |
| 2 | sidewalk |
| 3 | non-drivable-other |
| 4 | person |
| 5 | rider |
| 6 | 2-wheeler |
| 7 | small-vehicle |
| 8 | large-vehicle |
| 9 | barrier-solid |
| 10 | barrier-open |
| 11 | structures-sign |
| 12 | structures-pole |
| 13 | construction |
| 14 | vegetation |
| 15 | sky |

### Label3ID

Label3ID is a 26-class fine-granularity setup:

| ID | Class |
| ---: | --- |
| 0 | road |
| 1 | parking |
| 2 | sidewalk |
| 3 | rail track |
| 4 | person |
| 5 | rider |
| 6 | motorcycle |
| 7 | bicycle |
| 8 | autorickshaw |
| 9 | car |
| 10 | truck |
| 11 | bus |
| 12 | large-vehicle |
| 13 | curb |
| 14 | wall |
| 15 | fence |
| 16 | guard rail |
| 17 | billboard |
| 18 | traffic sign |
| 19 | traffic light |
| 20 | pole |
| 21 | obs-str-bar-fallback |
| 22 | building |
| 23 | bridge/tunnel |
| 24 | vegetation |
| 25 | sky |

## 5. Semantic Training Setup

The semantic dataset loader is `IDDSegDataset`. It performs:

- ImageNet normalization.
- Random resized crop to `512 x 512` during training.
- Horizontal flip augmentation.
- Color jitter augmentation.
- Validation resize to `512 x 512`.
- Ignore index `255`.

Common training settings across all projects:

| Setting | Value |
| --- | --- |
| Image size | `512 x 512` |
| Batch size | `32` |
| Workers | `8` |
| Seed | `42` |
| AMP | Enabled |
| Teacher epochs | `50` |
| Teacher LR | `1e-4` |
| Teacher weight decay | `1e-4` |
| Baseline epochs | `50` |
| Baseline LR | `6e-4` |
| Baseline weight decay | `1e-4` |
| Logit KD epochs | `125` |
| Logit KD LR | `6e-4` |
| Logit KD weight decay | `1e-4` |
| KD temperature | `4.0` |
| KD alpha | `1.0` |

The mask encoding can run in automatic mode. If the mask already contains contiguous training IDs, it is used directly; otherwise raw IDD IDs are remapped into the target training label space.

## 6. Model Families

### Original Teacher and Student

The original semantic project uses:

| Role | Backbone | Decoder | Channels |
| --- | --- | --- | --- |
| Teacher | `mobilenetv4_conv_large.e600_r384_in1k` | DeepLabV3+ | `[24, 48, 96, 192]` |
| Student | `mobilenetv4_conv_small.e2400_r224_in1k` | LR-ASPP | `[32, 32, 64, 96]` |

The Hailo-optimized LR-ASPP student uses deployment-friendly changes:

- Fixed interpolation sizes of `128 x 128` and `512 x 512`.
- `Sigmoid` instead of hard-sigmoid style operations.
- Export mode returns a raw tensor instead of a dictionary.

### ConvNeXt Teacher

The stronger teacher experiments use:

| Role | Backbone | Decoder | Channels |
| --- | --- | --- | --- |
| Teacher | ConvNeXt-Base | UPerNet | `[128, 256, 512, 1024]` |

The ConvNeXt UPerNet project uses:

| Setting | Value |
| --- | --- |
| Batch size | `24` |
| Teacher epochs | `50` |
| Logit KD epochs | `100` |

### DeepLab Student Variants

Two additional students were trained against the ConvNeXt teacher:

| Student |
| --- |
| MobileNetV4-L DeepLabV3+ |
| MobileNetV4-S DeepLabV3+ |

## 7. Logit Knowledge Distillation

Logit KD combines supervised cross-entropy with masked KL divergence between teacher and student output distributions:

```text
loss = CE(student, target) + alpha * KL(student_logits / T, teacher_logits / T)
```

where `T = 4.0` is the distillation temperature and `alpha = 1.0` is the distillation weight. This method is applied consistently across all label levels and all student architectures.

## 8. Semantic Training Results

The following values were recovered from the training logs. "Best mIoU" is the best validation mIoU recorded for that run.

### Original MobileNetV4 Teacher/Student Project

| Label | Run | Best mIoU | Best Epoch | Final Losses |
| --- | --- | ---: | ---: | --- |
| Label1 | Teacher | 78.12% | 50 | CE 0.1101 |
| Label1 | Baseline student | 72.42% | 49 | CE 0.1734 |
| Label1 | Logit KD | 72.15% | 123 | CE 0.1844, KL 0.4075, total 0.5919 |
| Label2 | Teacher | 67.61% | 42 | CE 0.1750 |
| Label2 | Baseline student | 59.34% | 46 | CE 0.2818 |
| Label2 | Logit KD | 59.67% | 123 | CE 0.2812, KL 0.4944, total 0.7756 |
| Label3 | Teacher | 60.91% | 50 | CE 0.1890 |
| Label3 | Baseline student | 51.56% | 47 | CE 0.3038 |
| Label3 | Logit KD | 51.38% | 114 | CE 0.3053, KL 0.5106, total 0.8158 |

Key observation: Logit KD provided modest improvement over the baseline for the original MobileNetV4-S LR-ASPP student. The teacher-student capacity gap remained the primary limiting factor.

### ConvNeXt UPerNet Teacher to MobileNetV4-S LR-ASPP Student

| Label | Run | Best mIoU | Best Epoch | Final Losses |
| --- | --- | ---: | ---: | --- |
| Label1 | Teacher | 81.92% | 39 | CE 0.0626 |
| Label1 | Logit KD | 71.85% | 67 | CE 0.2102, KL 0.8387, total 1.0489 |
| Label2 | Teacher | 73.14% | 42 | CE 0.0898 |
| Label2 | Logit KD | 59.37% | 92 | CE 0.3298, KL 1.2405, total 1.5702 |
| Label3 | Teacher | 68.54% | 44 | CE 0.0946 |
| Label3 | Logit KD | 51.59% | 87 | CE 0.3529, KL 1.2890, total 1.6419 |

Key observation: The ConvNeXt UPerNet teacher is materially stronger, but the compact LR-ASPP student remains capacity-limited and does not close much additional gap despite the stronger teacher signal.

### ConvNeXt Teacher to MobileNetV4-L DeepLabV3+ Student

| Label | Run | Best mIoU | Best Epoch | Final Losses |
| --- | --- | ---: | ---: | --- |
| Label1 | Teacher | 81.92% | 39 | same ConvNeXt teacher |
| Label1 | Logit KD | 78.87% | 116 | CE 0.0978, KL 0.2991, total 0.3969 |
| Label2 | Teacher | 73.14% | 42 | same ConvNeXt teacher |
| Label2 | Logit KD | 68.15% | 71 | CE 0.1716, KL 0.5660, total 0.7376 |

Key observation: MobileNetV4-L DeepLabV3+ closes much more of the teacher-student gap than the smaller LR-ASPP student, especially on Label1 and Label2. This is the highest-accuracy deployed student.

### ConvNeXt Teacher to MobileNetV4-S DeepLabV3+ Student

| Label | Run | Best mIoU | Best Epoch | Final Losses |
| --- | --- | ---: | ---: | --- |
| Label1 | Teacher | 81.92% | 39 | same ConvNeXt teacher |
| Label1 | Logit KD | 76.25% | 121 | CE 0.1451, KL 0.5144, total 0.6595 |
| Label2 | Teacher | 73.14% | 42 | same ConvNeXt teacher |
| Label2 | Logit KD | 63.99% | 95 | CE 0.2504, KL 0.8781, total 1.1285 |
| Label3 | Teacher | 68.54% | 44 | same ConvNeXt teacher |
| Label3 | Logit KD | 56.33% | 91 | CE 0.2717, KL 0.9373, total 1.2090 |

Key observation: MobileNetV4-S DeepLabV3+ is a strong middle ground — it materially improves over the original LR-ASPP student while remaining much smaller than the MobileNetV4-L variant.

## 9. YOLOv8n Instance Segmentation


The dataset bridge script creates a YOLO-compatible dataset at:

```text
datasets/idd_instance_seg
```

It hardlinks images and symlinks labels so Ultralytics can resolve images and segmentation labels from a local YOLO-style tree.

YOLO class mapping:

| YOLO ID | Class |
| ---: | --- |
| 0 | person_animal |
| 1 | rider |
| 2 | motorcycle_bicycle |
| 3 | autorickshaw_car |
| 4 | large_vehicle |

Training configuration:

| Setting | Value |
| --- | --- |
| Base model | `yolov8n-seg.pt` |
| Epochs | `100` |
| Image size | `512` |
| Batch size | `192` default |
| Workers | `8` |
| Device | CUDA device `0` |
| Project/name | `runs/idd_yolov8n_seg` |

YOLO dataset log summary:

| Split | Images | Backgrounds | Corrupt |
| --- | ---: | ---: | ---: |
| Train | 12,872 | 90 | 0 |
| Validation | 1,995 | 21 | 0 |

The training log reports 381 of 417 pretrained weights transferred from the base YOLOv8n-seg checkpoint.

Best and final YOLO metrics from `runs/idd_yolov8n_seg/results.csv`:

| Metric | Value |
| --- | ---: |
| Best box mAP50-95 | 0.21013 at epoch 90 |
| Best box mAP50 | 0.36054 |
| Best box precision | 0.69262 |
| Best box recall | 0.32409 |
| Best mask mAP50-95 | 0.15582 at epoch 88 |
| Best mask mAP50 | 0.32560 |
| Best mask precision | 0.63365 |
| Best mask recall | 0.28907 |
| Epoch 100 train box loss | 1.31612 |
| Epoch 100 train segmentation loss | 2.48142 |
| Epoch 100 train classification loss | 0.75843 |
| Epoch 100 train DFL loss | 0.92990 |
| Epoch 100 validation box loss | 1.35210 |
| Epoch 100 validation segmentation loss | 2.54123 |
| Epoch 100 validation classification loss | 0.78819 |
| Epoch 100 validation DFL loss | 0.93597 |
| Epoch 100 box mAP50-95 | 0.20966 |
| Epoch 100 mask mAP50-95 | 0.15497 |

## 10. Hailo Quantization and Compilation

The project exports selected PyTorch checkpoints to ONNX, parses them into Hailo HAR files, optimizes/quantizes with calibration images, and compiles HEF files for Hailo-8.

Common semantic Hailo flow:

1. Export PyTorch checkpoint to ONNX.
2. Parse ONNX to Hailo HAR.
3. Optimize HAR with a calibration set.
4. Compile optimized HAR to HEF for `hailo8`.
5. Copy HEFs into `Final__Demo`.

Common semantic model-script settings:

```text
normalization mean = [123.675, 116.28, 103.53]
normalization std  = [58.395, 57.12, 57.375]
calibration size   = 64 images/tensors
calibration batch  = 1
target             = hailo8
```

The Hailo logs show that optimization level was reduced to level 0 because only 64 calibration entries were available and no GPU was available for higher-level optimization. QAT fine-tuning was skipped.

### Semantic HEF Summary

| Hailo Folder | Model | ONNX Size | HEF Size | Compile FPS Estimate |
| --- | --- | ---: | ---: | ---: |
| `model1_hailo` | MobileNetV4-S LR-ASPP, 16 classes | 1.38 MB | 1.40 MB | 605.732 FPS post-allocation |
| `model2_hailo` | MobileNetV4-L DeepLabV3+, 16 classes | 25.59 MB | 6.92 MB | 51.6728 FPS post-allocation |
| `model3_hailo` | MobileNetV4-S DeepLabV3+, 16 classes | 10.63 MB | 10.41 MB | 121.596 FPS post-allocation |

Additional compile-log bottleneck FPS estimates:

| Model | Raw FPS |
| --- | ---: |
| `model1_hailo` | 728.753 |
| `model2_hailo` | 61.663 |
| `model3_hailo` | 138.300 |

### YOLOv8n-Seg HEF Summary

The YOLO Hailo folder is `yolov8n_seg_hailo`.

Export settings:

| Setting | Value |
| --- | --- |
| Source checkpoint task | Segment |
| Architecture | `yolov8n-seg` |
| Classes | 5 |
| ONNX opset | 11 |
| Dynamic axes | Disabled |
| NMS in export | Disabled |
| Half precision export | Disabled |
| Batch | 1 |
| Export device | CPU |

YOLO model-script settings:

```text
normalization([0, 0, 0], [255, 255, 255])
change_output_activation(conv45, sigmoid)
change_output_activation(conv61, sigmoid)
change_output_activation(conv74, sigmoid)
```

YOLO Hailo artifact summary:

| Artifact | Size |
| --- | ---: |
| ONNX | 13.20 MB |
| HEF | 7.41 MB |

YOLO compile log summary:

| Item | Value |
| --- | ---: |
| Calibration entries | 64 |
| Optimization level | Reduced to 0 |
| QAT | Skipped |
| Resolver FPS estimate | 224.836 |
| Context 1 post-allocation FPS | 1388.21 |
| Context 2 post-allocation FPS | 1732.73 |

## 11. Raspberry Pi Deployment

The deployment application is:

```text
Final__Demo/python_if_models_switch_pipelined.py
```

It uses:

- PyQt5 for the GUI.
- OpenCV for video/camera input and visualization.
- Hailo Platform APIs for loading HEFs and running inference.
- A thread-based inference loop to keep the UI responsive.

Deployed HEFs:

| File | Role |
| --- | --- |
| `model1_hailo.hef` | Semantic model 1, MobileNetV4-S LR-ASPP. |
| `model2_hailo.hef` | Semantic model 2, MobileNetV4-L DeepLabV3+. |
| `model3_hailo.hef` | Semantic model 3, MobileNetV4-S DeepLabV3+. |
| `yolov8n_seg_hailo.hef` | YOLOv8n-seg instance segmentation. |

Runtime constants:

| Setting | Value |
| --- | --- |
| Input size | `512` |
| Semantic classes | `16` |
| YOLO confidence threshold | `0.5` |
| YOLO NMS IoU threshold | `0.5` |
| YOLO DFL `REG_MAX` | `16` |
| YOLO mask dimension | `32` |

Semantic preprocessing:

1. Resize frame to `512 x 512`.
2. Convert BGR to RGB.
3. Convert to `float32`.
4. Run semantic HEF.
5. Apply `argmax` over class logits.
6. Map class IDs to a color palette.

YOLO preprocessing and decoding:

1. Letterbox input to `512 x 512`.
2. Convert BGR to RGB.
3. Run YOLO HEF.
4. Decode three detection heads:
   - stride 8: regression `conv44`, class `conv45`, mask `conv46`
   - stride 16: regression `conv60`, class `conv61`, mask `conv62`
   - stride 32: regression `conv73`, class `conv74`, mask `conv75`
   - prototype output: `conv48`
5. Decode boxes with DFL.
6. Apply confidence threshold and multiclass NMS.
7. Combine mask coefficients with prototype masks.
8. Undo letterbox scaling and overlay instance masks.

Application features:

- Load video files.
- Discover Raspberry Pi camera sources using libcamera/GStreamer and `/dev/video*`.
- Switch between semantic models at runtime.
- Enable or disable YOLO overlay.
- Adjust overlay alpha.
- Run a benchmark mode:

```text
python python_if_models_switch_pipelined.py --benchmark <video> [frames]
```

The benchmark prints average milliseconds per frame and FPS for each semantic model.

## 12. Model Selection Notes

The logs support the following practical conclusions:

- ConvNeXt UPerNet is the best teacher family, reaching 81.92% Label1, 73.14% Label2, and 68.54% Label3 mIoU.
- MobileNetV4-L DeepLabV3+ gives the best distilled student accuracy, with 78.87% Label1 and 68.15% Label2 using logit KD.
- MobileNetV4-S DeepLabV3+ is a strong compact deployment option, reaching 63.99% Label2 and 56.33% Label3 with logit KD — a material improvement over the original LR-ASPP student.
- Logit KD is consistently effective across all student architectures and label levels.
- Hailo compilation succeeded for all selected semantic and YOLO models, but the calibration and optimization setup was conservative — only 64 calibration samples were used and QAT was skipped.

## 13. Known Limitations

- The MobileNetV4-L DeepLabV3+ Label2 run was only recorded to epoch 17 for the KD-D variant (not used); the logit KD run completed fully.
- The Hailo logs indicate optimization level 0 due to limited calibration data and no GPU-assisted optimization. More calibration images and full optimization/QAT could improve quantized accuracy.
- YOLO mask mAP is modest but remains useful as a dynamic-object overlay when combined with semantic segmentation.


The deployable system is a hybrid semantic-plus-instance perception pipeline: semantic segmentation supplies dense road-scene layout, YOLO supplies dynamic object instances, and the Raspberry Pi application fuses both outputs in real time through the Hailo accelerator.

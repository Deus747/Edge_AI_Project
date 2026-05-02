import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from hailo_platform import (
    FormatType,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


INPUT_SIZE = 512
NUM_CLASSES = 16
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
REG_MAX = 16
MASK_DIM = 32
MODELS_OFF = "off"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

SEMANTIC_MODELS = {
    "model1": {"hef_path": "model1_hailo.hef", "to_rgb": True},
    "model2": {"hef_path": "model2_hailo.hef", "to_rgb": True},
    "model3": {"hef_path": "model3_hailo.hef", "to_rgb": True},
}

YOLO_HEF_PATH = "yolov8n_seg_hailo.hef"

CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]
SEMANTIC_PALETTE_RGB = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 60, 100],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

INSTANCE_CLASS_NAMES = [
    "person_animal",
    "rider",
    "motorcycle_bicycle",
    "autorickshaw_car",
    "large_vehicle",
]
INSTANCE_COLORS_BGR = np.array(
    [
        [60, 76, 231],
        [30, 180, 255],
        [60, 200, 60],
        [220, 120, 30],
        [180, 60, 180],
    ],
    dtype=np.uint8,
)

DETECTION_HEADS = [
    {"reg": "yolov8n_seg_hailo/conv44", "cls": "yolov8n_seg_hailo/conv45", "mask": "yolov8n_seg_hailo/conv46", "stride": 8},
    {"reg": "yolov8n_seg_hailo/conv60", "cls": "yolov8n_seg_hailo/conv61", "mask": "yolov8n_seg_hailo/conv62", "stride": 16},
    {"reg": "yolov8n_seg_hailo/conv73", "cls": "yolov8n_seg_hailo/conv74", "mask": "yolov8n_seg_hailo/conv75", "stride": 32},
]
PROTO_NAME = "yolov8n_seg_hailo/conv48"


def capture_reads_frame(source, api_preference):
    cap = cv2.VideoCapture(source, api_preference)
    if not cap.isOpened():
        cap.release()
        return False
    ret, frame = cap.read()
    cap.release()
    return bool(ret and frame is not None and frame.size > 0)


def find_camera_source():
    gst_pipeline = (
        "libcamerasrc ! "
        f"video/x-raw,format=RGB,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )
    if capture_reads_frame(gst_pipeline, cv2.CAP_GSTREAMER):
        return gst_pipeline, cv2.CAP_GSTREAMER, "Raspberry Pi Camera"

    video_devices = sorted(Path("/dev").glob("video*"))
    for device in video_devices:
        device_name = str(device)
        if capture_reads_frame(device_name, cv2.CAP_V4L2):
            return device_name, cv2.CAP_V4L2, device_name

    return None, None, None


def letterbox(image, new_shape=(INPUT_SIZE, INPUT_SIZE), color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def decode_dfl(reg_output):
    reg_output = reg_output.reshape(-1, 4, REG_MAX)
    distribution = softmax(reg_output, axis=-1)
    bins = np.arange(REG_MAX, dtype=np.float32)
    return np.sum(distribution * bins, axis=-1)


def decode_single_head(reg_output, cls_output, mask_output, stride):
    h, w, _ = cls_output.shape
    reg_dist = decode_dfl(reg_output)
    cls_scores = np.clip(cls_output.reshape(-1, 5), 0.0, 1.0)
    mask_coeffs = mask_output.reshape(-1, MASK_DIM)

    grid_y, grid_x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    anchor_points = np.stack((grid_x + 0.5, grid_y + 0.5), axis=-1).reshape(-1, 2)

    x1 = (anchor_points[:, 0] - reg_dist[:, 0]) * stride
    y1 = (anchor_points[:, 1] - reg_dist[:, 1]) * stride
    x2 = (anchor_points[:, 0] + reg_dist[:, 2]) * stride
    y2 = (anchor_points[:, 1] + reg_dist[:, 3]) * stride
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes, cls_scores, mask_coeffs


def clip_boxes_xyxy(boxes, width, height):
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter_area + 1e-6
    return inter_area / union


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    keep = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        sorted_indices = class_indices[np.argsort(scores[class_indices])[::-1]]
        while sorted_indices.size > 0:
            current = sorted_indices[0]
            keep.append(current)
            if sorted_indices.size == 1:
                break
            ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])
            sorted_indices = sorted_indices[1:][ious < iou_threshold]
    return np.array(keep, dtype=np.int32)


def crop_mask(mask, box):
    x1, y1, x2, y2 = box.astype(np.int32)
    cropped = np.zeros_like(mask, dtype=np.float32)
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def build_masks(proto, mask_coeffs, boxes, input_h, input_w):
    proto_h, proto_w, _ = proto.shape
    proto_flat = proto.reshape(-1, MASK_DIM)
    masks = sigmoid(mask_coeffs @ proto_flat.T).reshape(-1, proto_h, proto_w)

    scaled_boxes = boxes.copy().astype(np.float32)
    scaled_boxes[:, [0, 2]] *= proto_w / input_w
    scaled_boxes[:, [1, 3]] *= proto_h / input_h
    scaled_boxes = clip_boxes_xyxy(scaled_boxes, proto_w, proto_h)

    binary_masks = []
    for mask, box in zip(masks, scaled_boxes):
        cropped = crop_mask(mask, box)
        binary_masks.append(cropped > 0.5)
    return np.array(binary_masks, dtype=bool)


def scale_mask_from_letterbox(mask, pad):
    dw, dh = pad
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    right = int(round(dw + 0.1))
    bottom = int(round(dh + 0.1))
    network_mask = mask.astype(np.uint8) * 255
    network_mask = cv2.resize(network_mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    cropped = network_mask[top:INPUT_SIZE - bottom, left:INPUT_SIZE - right]
    resized = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized > 127


class SemanticPipeline:
    def __init__(self, device, name, hef_path, to_rgb):
        self.name = name
        self.to_rgb = to_rgb
        self.hef = HEF(hef_path)
        self.network_group = device.configure(self.hef)[0]
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.input_name = next(iter(self.input_vstreams_params.keys()))
        self.output_name = next(iter(self.output_vstreams_params.keys()))
        self._activation = None
        self._infer_streams = None
        self._opened = False

    def open(self):
        if self._opened:
            return
        self._activation = self.network_group.activate()
        self._activation.__enter__()
        self._infer_streams = InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        )
        self._infer_streams.__enter__()
        self._opened = True

    def close(self):
        if not self._opened:
            return
        self._infer_streams.__exit__(None, None, None)
        self._activation.__exit__(None, None, None)
        self._infer_streams = None
        self._activation = None
        self._opened = False

    def preprocess(self, image_bgr):
        base_bgr = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB) if self.to_rgb else base_bgr.copy()
        tensor = tensor.astype(np.float32)
        return base_bgr, tensor

    def run_inference(self, tensor):
        self.open()
        return self._infer_streams.infer({self.input_name: np.expand_dims(tensor, axis=0)})

    def postprocess(self, outputs):
        raw_output = outputs[self.output_name][0]
        mask = np.argmax(raw_output, axis=-1).astype(np.uint8)
        if mask.shape != (INPUT_SIZE, INPUT_SIZE):
            mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        color_mask_bgr = cv2.cvtColor(SEMANTIC_PALETTE_RGB[mask], cv2.COLOR_RGB2BGR)
        return mask, color_mask_bgr

    def infer(self, image_bgr):
        base_bgr, tensor = self.preprocess(image_bgr)
        outputs = self.run_inference(tensor)
        mask, color_mask_bgr = self.postprocess(outputs)
        return base_bgr, mask, color_mask_bgr


class YoloPipeline:
    def __init__(self, device):
        self.hef = HEF(YOLO_HEF_PATH)
        self.network_group = device.configure(self.hef)[0]
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.input_name = next(iter(self.input_vstreams_params.keys()))
        self._activation = None
        self._infer_streams = None
        self._opened = False

    def open(self):
        if self._opened:
            return
        self._activation = self.network_group.activate()
        self._activation.__enter__()
        self._infer_streams = InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        )
        self._infer_streams.__enter__()
        self._opened = True

    def close(self):
        if not self._opened:
            return
        self._infer_streams.__exit__(None, None, None)
        self._activation.__exit__(None, None, None)
        self._infer_streams = None
        self._activation = None
        self._opened = False

    def preprocess(self, image_bgr):
        input_bgr, _, pad = letterbox(image_bgr, (INPUT_SIZE, INPUT_SIZE))
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        network_input = input_rgb.astype(np.float32)
        return network_input, pad

    def run_inference(self, network_input):
        self.open()
        return self._infer_streams.infer({self.input_name: np.expand_dims(network_input, axis=0)})

    def postprocess(self, outputs, pad):
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_mask_coeffs = []

        for head in DETECTION_HEADS:
            boxes, cls_scores, mask_coeffs = decode_single_head(
                outputs[head["reg"]][0],
                outputs[head["cls"]][0],
                outputs[head["mask"]][0],
                head["stride"],
            )
            class_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
            keep = scores >= CONF_THRESHOLD
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_class_ids.append(class_ids[keep])
            all_mask_coeffs.append(mask_coeffs[keep])

        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=np.float32)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
        class_ids = np.concatenate(all_class_ids, axis=0) if all_class_ids else np.empty((0,), dtype=np.int32)
        mask_coeffs = np.concatenate(all_mask_coeffs, axis=0) if all_mask_coeffs else np.empty((0, MASK_DIM), dtype=np.float32)

        if boxes.shape[0] == 0:
            return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        boxes = clip_boxes_xyxy(boxes, INPUT_SIZE, INPUT_SIZE)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        mask_coeffs = mask_coeffs[valid]

        if boxes.shape[0] == 0:
            return [], scores, class_ids

        keep = multiclass_nms(boxes, scores, class_ids, NMS_IOU_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        proto = outputs[PROTO_NAME][0]
        masks_128 = build_masks(proto, mask_coeffs, boxes, INPUT_SIZE, INPUT_SIZE)
        masks_512 = [scale_mask_from_letterbox(mask, pad) for mask in masks_128]
        return masks_512, scores, class_ids

    def infer(self, image_bgr):
        network_input, pad = self.preprocess(image_bgr)
        outputs = self.run_inference(network_input)
        return self.postprocess(outputs, pad)


def draw_name_tags(frame_bgr, masks, scores, class_ids):
    output = frame_bgr.copy()
    for mask, score, class_id in zip(masks, scores, class_ids):
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        x = int(xs.min())
        y = int(max(14, ys.min()))
        color = INSTANCE_COLORS_BGR[int(class_id)].tolist()
        label = f"{INSTANCE_CLASS_NAMES[int(class_id)]} {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_top = max(0, y - label_h - 6)
        label_bottom = min(output.shape[0] - 1, y + 2)
        cv2.rectangle(output, (x, label_top), (x + label_w + 6, label_bottom), color, -1)
        cv2.putText(
            output,
            label,
            (x + 3, max(12, label_bottom - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


class InferenceThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    stats_signal = pyqtSignal(str)

    def __init__(self, source, api_preference=cv2.CAP_ANY, source_label="Video"):
        super().__init__()
        self.source = source
        self.api_preference = api_preference
        self.source_label = source_label
        self._run_flag = True
        self.active_model = "model1"
        self.yolo_enabled = True
        self.alpha = 0.45
        self._last_frame_bgr = None
        self._opened_semantic_model = None
        self._device_context = None
        self._device = None
        self._semantic_pipelines = None
        self._yolo_pipeline = None

    def set_active_model(self, model_name):
        self.active_model = model_name

    def set_yolo_enabled(self, enabled):
        self.yolo_enabled = enabled

    def set_alpha(self, alpha):
        self.alpha = alpha

    def ensure_pipelines(self):
        if self._semantic_pipelines is not None and self._yolo_pipeline is not None:
            return self._semantic_pipelines, self._yolo_pipeline
        self._device_context = VDevice()
        self._device = self._device_context.__enter__()
        self._semantic_pipelines = {
            name: SemanticPipeline(self._device, name, cfg["hef_path"], cfg["to_rgb"])
            for name, cfg in SEMANTIC_MODELS.items()
        }
        self._yolo_pipeline = YoloPipeline(self._device)
        return self._semantic_pipelines, self._yolo_pipeline

    def close_pipelines(self):
        if self._semantic_pipelines is not None:
            for pipeline in self._semantic_pipelines.values():
                pipeline.close()
        if self._yolo_pipeline is not None:
            self._yolo_pipeline.close()
        self._semantic_pipelines = None
        self._yolo_pipeline = None
        self._device = None
        self._opened_semantic_model = None
        if self._device_context is not None:
            self._device_context.__exit__(None, None, None)
            self._device_context = None

    def _select_semantic_pipeline(self, semantic_pipelines, active_model):
        if self._opened_semantic_model == active_model:
            return semantic_pipelines[active_model]
        if self._opened_semantic_model is not None:
            semantic_pipelines[self._opened_semantic_model].close()
        pipeline = semantic_pipelines[active_model]
        pipeline.open()
        self._opened_semantic_model = active_model
        return pipeline

    def process_frame(self, frame, semantic_pipelines, yolo_pipeline, executor, active_model=None, yolo_enabled=None, alpha=None):
        active_model = self.active_model if active_model is None else active_model
        yolo_enabled = self.yolo_enabled if yolo_enabled is None else yolo_enabled
        alpha = self.alpha if alpha is None else alpha

        start = time.perf_counter()
        if active_model == MODELS_OFF:
            if semantic_pipelines is not None and self._opened_semantic_model is not None:
                semantic_pipelines[self._opened_semantic_model].close()
                self._opened_semantic_model = None
            output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed_ms = (time.perf_counter() - start) * 1000
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            status = f"Source: {self.source_label} | Model: Off | YOLO: Off | Frame: {elapsed_ms:.1f} ms | FPS: {fps:.1f}"
            return output_frame, status

        semantic_pipeline = self._select_semantic_pipeline(semantic_pipelines, active_model)

        semantic_preprocess_future = executor.submit(semantic_pipeline.preprocess, frame)
        yolo_preprocess_future = executor.submit(yolo_pipeline.preprocess, frame) if yolo_enabled else None

        base_bgr, semantic_tensor = semantic_preprocess_future.result()
        if yolo_enabled:
            yolo_tensor, yolo_pad = yolo_preprocess_future.result()

        semantic_outputs = semantic_pipeline.run_inference(semantic_tensor)
        semantic_pipeline.close()
        self._opened_semantic_model = None
        semantic_postprocess_future = executor.submit(semantic_pipeline.postprocess, semantic_outputs)

        yolo_count = 0
        masks = []
        scores = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)
        if yolo_enabled:
            yolo_outputs = yolo_pipeline.run_inference(yolo_tensor)
            yolo_pipeline.close()
            yolo_postprocess_future = executor.submit(yolo_pipeline.postprocess, yolo_outputs, yolo_pad)

        _, semantic_color_bgr = semantic_postprocess_future.result()
        if yolo_enabled:
            masks, scores, class_ids = yolo_postprocess_future.result()
            yolo_count = len(scores)

        infer_ms = (time.perf_counter() - start) * 1000

        render_start = time.perf_counter()
        fused_color_bgr = semantic_color_bgr.copy()
        if yolo_enabled:
            for mask, class_id in zip(masks, class_ids):
                fused_color_bgr[mask] = INSTANCE_COLORS_BGR[int(class_id)]
            fused_overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, fused_color_bgr, alpha, 0)
            fused_overlay = draw_name_tags(fused_overlay, masks, scores, class_ids)
        else:
            fused_overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, fused_color_bgr, alpha, 0)
        render_ms = (time.perf_counter() - render_start) * 1000

        total_ms = infer_ms + render_ms
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        status = (
            f"Source: {self.source_label} | Model: {active_model} | YOLO: {'On' if yolo_enabled else 'Off'} | "
            f"Alpha: {alpha:.2f} | Pipelined infer: {infer_ms:.1f} ms | Render: {render_ms:.1f} ms | FPS: {fps:.1f} | "
            f"Instances: {yolo_count}"
        )
        return cv2.cvtColor(fused_overlay, cv2.COLOR_BGR2RGB), status

    def run(self):
        cap = cv2.VideoCapture(self.source, self.api_preference)
        if not cap.isOpened():
            self.stats_signal.emit(f"Could not open source: {self.source_label}")
            return

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                while self._run_flag:
                    ret, frame = cap.read()
                    if ret:
                        self._last_frame_bgr = frame.copy()
                    elif self._last_frame_bgr is None:
                        break

                    source_frame = self._last_frame_bgr
                    if source_frame is None:
                        break

                    active_model = self.active_model
                    yolo_enabled = self.yolo_enabled
                    alpha = self.alpha

                    if active_model == MODELS_OFF:
                        self.close_pipelines()
                        semantic_pipelines = None
                        yolo_pipeline = None
                    else:
                        semantic_pipelines, yolo_pipeline = self.ensure_pipelines()

                    output_frame, status = self.process_frame(
                        source_frame,
                        semantic_pipelines,
                        yolo_pipeline,
                        executor,
                        active_model,
                        yolo_enabled,
                        alpha,
                    )
                    self.change_pixmap_signal.emit(output_frame)
                    self.stats_signal.emit(status)

                    if not ret:
                        break
        finally:
            self.close_pipelines()

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Switch + YOLO Overlay")
        self.setFixedSize(680, 760)

        self.active_model = "model1"
        self.yolo_enabled = True
        self.alpha = 0.45
        self.thread = None
        self.model_buttons = {}
        self.video_path = None
        self.using_camera = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Select a video to start")
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.layout.addWidget(self.image_label)

        self.stats_label = QLabel("Model: - | YOLO: - | Infer: 0 ms | Render: 0 ms | FPS: 0")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.stats_label)

        controls = QHBoxLayout()
        self.layout.addLayout(controls)

        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self.open_video)
        controls.addWidget(self.btn_open)

        self.btn_source_toggle = QPushButton("Use Camera")
        self.btn_source_toggle.setCheckable(True)
        self.btn_source_toggle.clicked.connect(self.toggle_source)
        controls.addWidget(self.btn_source_toggle)

        for model_name in [MODELS_OFF, "model1", "model2", "model3"]:
            button = QPushButton(model_name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=model_name: self.set_active_model(name))
            controls.addWidget(button)
            self.model_buttons[model_name] = button

        self.btn_yolo = QPushButton("YOLO On")
        self.btn_yolo.setCheckable(True)
        self.btn_yolo.setChecked(True)
        self.btn_yolo.clicked.connect(self.toggle_yolo)
        controls.addWidget(self.btn_yolo)

        alpha_row = QHBoxLayout()
        self.layout.addLayout(alpha_row)
        self.alpha_label = QLabel(f"Mask Alpha: {self.alpha:.2f}")
        alpha_row.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.set_alpha)
        alpha_row.addWidget(self.alpha_slider)

        self.update_model_buttons()

    def update_model_buttons(self):
        for model_name, button in self.model_buttons.items():
            button.setChecked(model_name == self.active_model)

    def set_active_model(self, model_name):
        self.active_model = model_name
        self.update_model_buttons()
        if self.thread and self.thread.isRunning():
            self.thread.set_active_model(model_name)

    def toggle_yolo(self):
        self.yolo_enabled = self.btn_yolo.isChecked()
        self.btn_yolo.setText("YOLO On" if self.yolo_enabled else "YOLO Off")
        if self.thread and self.thread.isRunning():
            self.thread.set_yolo_enabled(self.yolo_enabled)

    def set_alpha(self, value):
        self.alpha = value / 100.0
        self.alpha_label.setText(f"Mask Alpha: {self.alpha:.2f}")
        if self.thread and self.thread.isRunning():
            self.thread.set_alpha(self.alpha)

    def start_source(self, source, api_preference=cv2.CAP_ANY, source_label="Video"):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

        self.thread = InferenceThread(source, api_preference, source_label)
        self.thread.set_active_model(self.active_model)
        self.thread.set_yolo_enabled(self.yolo_enabled)
        self.thread.set_alpha(self.alpha)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.start()

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not file_name:
            return
        self.video_path = file_name
        self.using_camera = False
        self.btn_source_toggle.setChecked(False)
        self.btn_source_toggle.setText("Use Camera")
        self.start_source(file_name, cv2.CAP_ANY, "Video")

    def use_opened_video(self):
        if not self.video_path:
            self.update_stats("No video selected. Click Open Video first.")
            self.using_camera = False
            self.btn_source_toggle.setChecked(False)
            self.btn_source_toggle.setText("Use Camera")
            return
        self.using_camera = False
        self.btn_source_toggle.setChecked(False)
        self.btn_source_toggle.setText("Use Camera")
        self.start_source(self.video_path, cv2.CAP_ANY, "Video")

    def use_camera(self):
        source, api_preference, source_label = find_camera_source()
        if source is None:
            self.update_stats("Camera not found: no /dev/video* or usable libcamera GStreamer source")
            self.btn_source_toggle.setChecked(False)
            self.btn_source_toggle.setText("Use Camera")
            return
        self.using_camera = True
        self.btn_source_toggle.setChecked(True)
        self.btn_source_toggle.setText("Use Video")
        self.start_source(source, api_preference, source_label)

    def toggle_source(self):
        if self.btn_source_toggle.isChecked():
            self.use_camera()
        else:
            self.use_opened_video()

    def update_image(self, rgb_image):
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_format.scaled(600, 600, Qt.KeepAspectRatio))
        self.image_label.setPixmap(pixmap)

    def update_stats(self, text):
        self.stats_label.setText(text)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


def benchmark_video(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    with VDevice() as device:
        semantic_pipelines = {
            name: SemanticPipeline(device, name, cfg["hef_path"], cfg["to_rgb"])
            for name, cfg in SEMANTIC_MODELS.items()
        }
        yolo_pipeline = YoloPipeline(device)
        worker = InferenceThread(video_path)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for model_name in SEMANTIC_MODELS:
                    worker.active_model = model_name
                    worker._opened_semantic_model = None
                    frame_times = []
                    processed = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    while processed < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_start = time.perf_counter()
                        worker.process_frame(frame, semantic_pipelines, yolo_pipeline, executor)
                        frame_times.append((time.perf_counter() - frame_start) * 1000)
                        processed += 1

                    if frame_times:
                        avg_ms = float(np.mean(frame_times))
                        print(f"[*] {model_name}: {avg_ms:.2f} ms/frame, {1000.0 / avg_ms:.2f} FPS over {processed} frames")
                    else:
                        print(f"[*] {model_name}: no frames processed")
        finally:
            for pipeline in semantic_pipelines.values():
                pipeline.close()
            yolo_pipeline.close()
            cap.release()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--benchmark":
        frame_count = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
        benchmark_video(sys.argv[2], frame_count)
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

import os

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset roots
DATA_ROOT = os.environ.get(
    "IDD_DATA_ROOT",
    os.path.dirname(PROJECT_DIR),  # sibling IDDL1/IDDL2/IDDL3 by default
)
LABEL_LEVEL = os.environ.get("IDD_LABEL_LEVEL", "Label2ID")

LEVEL_TO_DIR = {
    "Label1ID": "IDDL1",
    "Label2ID": "IDDL2",
    "Label3ID": "IDDL3",
}

IMG_BASE = os.path.join(DATA_ROOT, LEVEL_TO_DIR[LABEL_LEVEL])
TRAIN_IMG_SRC = os.path.join(IMG_BASE, "train", "images")
VAL_IMG_SRC = os.path.join(IMG_BASE, "val", "images")

# Instance labels (YOLO txt polygons)
INSTANCE_LABELS_ROOT = os.environ.get(
    "IDD_INSTANCE_LABELS_ROOT",
    os.path.join(os.path.dirname(PROJECT_DIR), "Instance_Segmentation"),
)
TRAIN_LABEL_SRC = os.path.join(INSTANCE_LABELS_ROOT, "train")
VAL_LABEL_SRC = os.path.join(INSTANCE_LABELS_ROOT, "val")

# YOLO dataset bridge (symlink tree)
BRIDGE_ROOT = os.path.join(PROJECT_DIR, "datasets", "idd_instance_seg")
BRIDGE_IMG_TRAIN = os.path.join(BRIDGE_ROOT, "images", "train")
BRIDGE_IMG_VAL = os.path.join(BRIDGE_ROOT, "images", "val")
BRIDGE_LBL_TRAIN = os.path.join(BRIDGE_ROOT, "labels", "train")
BRIDGE_LBL_VAL = os.path.join(BRIDGE_ROOT, "labels", "val")

DATA_YAML = os.path.join(PROJECT_DIR, "idd_instance_seg.yaml")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# IDD Level2 instance groups used by provided labels
NAMES = {
    0: "person_animal",
    1: "rider",
    2: "motorcycle_bicycle",
    3: "autorickshaw_car",
    4: "large_vehicle",
}

# Training defaults
MODEL = "yolov8n-seg.pt"
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "100"))
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "512"))
BATCH = int(os.environ.get("YOLO_BATCH", "192"))
WORKERS = int(os.environ.get("YOLO_WORKERS", "8"))
DEVICE = os.environ.get("YOLO_DEVICE", "0")
PROJECT_NAME = "idd_yolov8n_seg"

# ============================================================
# config.py — Shared configuration
#
# Label level is controlled via environment variable:
#   export IDD_LABEL_LEVEL=Label1ID   # Label1ID | Label2ID | Label3ID
#
# Data root defaults to the parent of this project directory.
# Override with:
#   export IDD_DATA_ROOT=/path/to/folder/containing/IDDL1_IDDL2_IDDL3
# ============================================================

import os
import torch

# ------------------------------------------------------------------
# Label level — set via env var, default Label1ID
# ------------------------------------------------------------------
LABEL_LEVEL = os.environ.get('IDD_LABEL_LEVEL', 'Label1ID')

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.environ.get('IDD_DATA_ROOT',
                               os.path.dirname(_PROJECT_DIR))

CKPT_DIR     = os.path.join(_PROJECT_DIR, 'checkpoints', LABEL_LEVEL)
PLOT_DIR     = os.path.join(_PROJECT_DIR, 'plots',       LABEL_LEVEL)
LOG_DIR      = os.path.join(_PROJECT_DIR, 'logs',        LABEL_LEVEL)
_DEFAULT_TEACHER_PROJECT = os.path.join(
    os.path.dirname(_PROJECT_DIR), 'idd_kd_project_convnext_upernet'
)
TEACHER_PROJECT_DIR = os.environ.get('IDD_TEACHER_PROJECT_DIR', _DEFAULT_TEACHER_PROJECT)
TEACHER_CKPT = os.path.join(TEACHER_PROJECT_DIR, 'checkpoints', LABEL_LEVEL, 'teacher_best.pth')

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ------------------------------------------------------------------
# Per-level config table
# ------------------------------------------------------------------
_LEVEL_CFG = {
    'Label1ID': {
        'num_classes': 7,
        'data_dir'   : 'IDDL1',
        'class_names': [
            'road', 'sky', 'vegetation', 'building',
            'vehicle', 'person', 'background',
        ],
        'palette': {
            0: (128,  64, 128),   # road
            1: ( 70, 130, 180),   # sky
            2: (107, 142,  35),   # vegetation
            3: ( 70,  70,  70),   # building
            4: (  0,   0, 142),   # vehicle
            5: (220,  20,  60),   # person
            6: (  0,   0,   0),   # background
        },
    },

    'Label2ID': {
        'num_classes': 16,
        'data_dir'   : 'IDDL2',
        # 16 contiguous classes derived from level2Id in label.py (ids 0-15)
        'class_names': [
            'road',            # 0
            'drivable-other',  # 1  (parking + drivable fallback)
            'sidewalk',        # 2
            'non-drivable-other', # 3 (rail track + non-drivable fallback)
            'person',          # 4  (person + animal)
            'rider',           # 5
            '2-wheeler',       # 6  (motorcycle + bicycle)
            'small-vehicle',   # 7  (autorickshaw + car — merged at L2)
            'large-vehicle',   # 8  (truck/bus/caravan/trailer/train/fallback)
            'barrier-solid',   # 9  (curb + wall)
            'barrier-open',    # 10 (fence + guard rail)
            'structures-sign', # 11 (billboard + traffic sign + traffic light)
            'structures-pole', # 12 (pole + polegroup + obs-str-bar-fallback)
            'construction',    # 13 (building + bridge + tunnel)
            'vegetation',      # 14
            'sky',             # 15 (sky + fallback background)
        ],
        'palette': {
            0:  (128,  64, 128),   # road
            1:  (250, 170, 160),   # drivable-other
            2:  (244,  35, 232),   # sidewalk
            3:  (230, 150, 140),   # non-drivable-other
            4:  (220,  20,  60),   # person
            5:  (255,   0,   0),   # rider
            6:  (  0,   0, 230),   # 2-wheeler
            7:  (  0,   0, 142),   # small-vehicle
            8:  (  0,   0,  70),   # large-vehicle
            9:  (220, 190,  40),   # barrier-solid
            10: (190, 153, 153),   # barrier-open
            11: (220, 220,   0),   # structures-sign
            12: (153, 153, 153),   # structures-pole
            13: ( 70,  70,  70),   # construction
            14: (107, 142,  35),   # vegetation
            15: ( 70, 130, 180),   # sky
        },
    },

    'Label3ID': {
        'num_classes': 26,
        'data_dir'   : 'IDDL3',
        'class_names': [
            'road', 'parking', 'sidewalk', 'rail track',
            'person', 'rider', 'motorcycle', 'bicycle',
            'autorickshaw', 'car', 'truck', 'bus',
            'large-vehicle', 'curb', 'wall', 'fence',
            'guard rail', 'billboard', 'traffic sign', 'traffic light',
            'pole', 'obs-str-bar-fallback', 'building', 'bridge/tunnel',
            'vegetation', 'sky',
        ],
        'palette': {
             0: (128,  64, 128),
             1: (250, 170, 160),
             2: (244,  35, 232),
             3: (230, 150, 140),
             4: (220,  20,  60),
             5: (255,   0,   0),
             6: (  0,   0, 230),
             7: (119,  11,  32),
             8: (255, 204,  54),
             9: (  0,   0, 142),
            10: (  0,   0,  70),
            11: (  0,  60, 100),
            12: (136, 143, 153),
            13: (220, 190,  40),
            14: (102, 102, 156),
            15: (190, 153, 153),
            16: (180, 165, 180),
            17: (174,  64,  67),
            18: (220, 220,   0),
            19: (250, 170,  30),
            20: (153, 153, 153),
            21: (169, 187, 214),
            22: ( 70,  70,  70),
            23: (150, 100, 100),
            24: (107, 142,  35),
            25: ( 70, 130, 180),
        },
    },
}

_cfg = _LEVEL_CFG[LABEL_LEVEL]

NUM_CLASSES  = _cfg['num_classes']
CLASS_NAMES  = _cfg['class_names']
PALETTE      = _cfg['palette']

_data_dir    = os.path.join(DATA_ROOT, _cfg['data_dir'])

# ------------------------------------------------------------------
# Dataset split paths
# ------------------------------------------------------------------
TRAIN_IMG  = os.path.join(_data_dir, 'train', 'images')
TRAIN_MASK = os.path.join(_data_dir, 'train', 'masks')
VAL_IMG    = os.path.join(_data_dir, 'val',   'images')
VAL_MASK   = os.path.join(_data_dir, 'val',   'masks')
TEST_IMG   = os.path.join(_data_dir, 'test',  'images')
TEST_MASK  = os.path.join(_data_dir, 'test',  'masks')

IGNORE_INDEX = 255
IMG_SIZE     = (512, 512)

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
BATCH_SIZE  = 32
NUM_WORKERS = 8
SEED        = 42
AMP         = True
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# KD training (shared across all strategies)
KD_EPOCHS = 100
KD_LR     = 6e-4
KD_WD     = 1e-4

# ------------------------------------------------------------------
# KD loss weights
# ------------------------------------------------------------------
TEMPERATURE = 4.0
ALPHA       = 1.0    # logit KD weight            (all strategies)
BETA_FEAT   = 0.05    # feature mimicking weight    (Strategy B)
BETA_CWD    = 1.5    # channel-wise distill weight (Strategy D)
BETA_ENC_D  = 0.1    # encoder feature KD weight   (Strategy D)
BETA_DEC_D  = 0.01   # decoder feature KD weight   (Strategy D)
BETA_STRUCTURAL = 0.5  # structural edge KD weight (Strategy Struct)

# ------------------------------------------------------------------
# Adaptive auxiliary-loss weighting (KD-D / KD-Struct)
# ------------------------------------------------------------------
AUTO_BETA_ENABLED      = True
AUTO_BETA_WARMUP_STEPS = 1500     # roughly first epoch at common batch sizes
AUTO_BETA_EMA_MOM      = 0.98
AUTO_BETA_MIN          = 1e-4
AUTO_BETA_MAX          = 500.0
AUTO_TARGET_ENC        = 0.05     # target contribution to total loss
AUTO_TARGET_DEC        = 0.05
AUTO_TARGET_STRUCT     = 0.02

# ------------------------------------------------------------------
# Architecture channel dims
# Run models/probe.py to verify for your timm version
# ------------------------------------------------------------------
TEACHER_BB_CHANNELS = [128, 256, 512, 1024]   # ConvNeXt-Base C2-C5
STUDENT_BB_CHANNELS = [32,  32,  64,  96]   # MobileNetV4-S S1-S4

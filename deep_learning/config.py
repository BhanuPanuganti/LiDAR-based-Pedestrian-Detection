import os

# =============================
# BASE PATH
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================
# CUSTOM OPENPCDET ROOT
# =============================
OPENPCDET_ROOT = os.path.join(BASE_DIR, "custom_openpcdet")


# =============================
# KITTI DATA PATHS (for classical pipeline)
# =============================

TRAIN_ROOT   = os.path.join(OPENPCDET_ROOT, "data", "kitti", "training")

VELODYNE_DIR = os.path.join(TRAIN_ROOT, "velodyne")
LABEL_DIR    = os.path.join(TRAIN_ROOT, "label_2")
CALIB_DIR    = os.path.join(TRAIN_ROOT, "calib")
IMAGE_DIR    = os.path.join(TRAIN_ROOT, "image_2")


# =============================
# CONFIG FILES
# =============================
CONFIG_DIR = os.path.join(BASE_DIR, "deep_learning", "configs")

POINTPILLAR_PED_CFG = os.path.join(CONFIG_DIR, "pointpillar_ped.yaml")
POINTPILLAR_CFG     = os.path.join(CONFIG_DIR, "pointpillar.yaml")
SECOND_CFG          = os.path.join(CONFIG_DIR, "second.yaml")

# =============================
# CHECKPOINTS (YOU MOVED THESE)
# =============================
PTH_DIR = os.path.join(BASE_DIR, "deep_learning", "pretrained", "pth_files")

POINTPILLAR_15_CKPT = os.path.join(
    BASE_DIR,
    "custom_openpcdet",
    "output",
    "kitti_models",
    "pointpillar_ped",
    "exp_A_15pct",
    "ckpt",
    "checkpoint_epoch_25.pth"
)

POINTPILLAR_PRETRAINED_CKPT = os.path.join(PTH_DIR, "pointpillar_7728.pth")
SECOND_PRETRAINED_CKPT      = os.path.join(PTH_DIR, "second_7862.pth")

# =============================
# SAMPLE FILE
# =============================
SAMPLE_FILE = os.path.join(
    OPENPCDET_ROOT,
    "data",
    "kitti",
    "training",
    "velodyne",
    "000008.bin"
)

# =============================
# EXPERIMENT TAGS
# =============================
EXP_TAG_PP_15 = "exp_A_15pct"
EXP_TAG_SEC   = "exp_C_second"

# =============================
# LEGACY-COMPAT CONSTANTS
# =============================
# These names are referenced by older helper scripts.
TOOLS_DIR = os.path.join(OPENPCDET_ROOT, "tools")
CONFIG_FILE = os.path.join(TOOLS_DIR, "cfgs", "pointpillar_ped.yaml")
EXP_TAG = EXP_TAG_PP_15

# Default experiment directory for training-log parsing helpers.
EXP_DIR = os.path.join(OPENPCDET_ROOT, "output", "kitti_models", "pointpillar_ped", EXP_TAG_PP_15)
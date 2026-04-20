import os

# Root of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If config.py is inside a subfolder, adjust:
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================
# PATHS
# =============================

OPENPCDET_ROOT = os.path.join(BASE_DIR, "custom_openpcdet")

KITTI_ROOT = os.path.join(OPENPCDET_ROOT, "data", "kitti")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Derived paths
TRAIN_ROOT   = os.path.join(KITTI_ROOT, 'training')
VELODYNE_DIR = os.path.join(TRAIN_ROOT, 'velodyne')
LABEL_DIR    = os.path.join(TRAIN_ROOT, 'label_2')
CALIB_DIR    = os.path.join(TRAIN_ROOT, 'calib')
IMAGE_DIR    = os.path.join(TRAIN_ROOT, 'image_2')

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")
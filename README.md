# LiDAR Pedestrian Detection Project

This repository combines two complementary approaches for pedestrian detection from LiDAR data:

1. Classical machine learning pipeline (feature engineering + XGBoost)
2. Deep learning pipeline using a local OpenPCDet fork in the folder custom_openpcdet

The project is organized so a new user can run everything from one command, generate evaluation outputs, and save visualization files.

---

## 1. What This Project Does

The project supports:

- Data inspection and bird's-eye-view visualization from KITTI point clouds
- Classical ML training and evaluation with saved metrics and plots
- Pretrained PointPillars evaluation
- Pretrained SECOND evaluation
- Consolidated visualization export to outputs/visuals

---

## 2. Repository Layout (Important Files)

Top-level:

- config.py: Global project paths for the classical pipeline
- setup_env.py: Windows-safe environment helper (Torch DLL + SharedArray stub)
- main.py: Initial data sanity run and BEV image save
- main_classical_ml.py: End-to-end classical ML pipeline
- main_deep_learning.py: Training-log plotting helper
- run_all_pipeline.ps1: One-click orchestrator for full workflow
- requirements.txt: Python dependencies
- .gitignore: Git ignore rules for outputs/artifacts

Core folders:

- classical_pipeline: Classical detection pipeline modules
  - classical_pipeline/dataset.py
  - classical_pipeline/features.py
  - classical_pipeline/model.py
  - classical_pipeline/evaluate.py
- data_utils: KITTI loaders and calibration helpers
- visualization: BEV plotting utility
  - visualization/bev.py
- deep_learning: Deep-learning wrappers and pretrained runners
  - deep_learning/config.py
  - deep_learning/pretrained/pointpillers/evaluate_pretrained.py
  - deep_learning/pretrained/second/evaluate_pretrained.py
  - deep_learning/pretrained/second/evaluate.py
  - deep_learning/pretrained/second/inference.py
  - deep_learning/pretrained/pointpillers/visualize.py
  - deep_learning/pretrained/second/visualize_metrics.py
  - deep_learning/pointpillers_15/visualize.py
- custom_openpcdet: Local OpenPCDet codebase used by runners
- scripts: Utility scripts
  - scripts/save_all_visualizations.py

Generated outputs:

- outputs/results: Classical pipeline artifacts
- outputs/visuals: Saved image outputs
- custom_openpcdet/output: OpenPCDet train/eval logs and result files

---

## 3. Data Prerequisites

Expected KITTI location inside this repository:

custom_openpcdet/data/kitti

Expected substructure:

- custom_openpcdet/data/kitti/training/velodyne
- custom_openpcdet/data/kitti/training/label_2
- custom_openpcdet/data/kitti/training/calib
- custom_openpcdet/data/kitti/ImageSets
- custom_openpcdet/data/kitti/kitti_infos_*.pkl

Notes:

- Evaluation relies on info pickle files such as kitti_infos_val.pkl.
- If dataset is reported empty, verify data path and info files first.

---

## 4. Environment Setup

### 4.1 Create and Activate a Virtual Environment (recommended)

Windows PowerShell example:

```powershell
cd C:\BTP\Additional\lidar-pedestrian-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.2 Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.3 Install CUDA-Compatible PyTorch

Pick the command matching your CUDA runtime, for example CUDA 11.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.4 Install spconv Matching Torch/CUDA

Install only one compatible wheel, for example:

```powershell
pip install spconv-cu118
```

---

## 5. One-Command Workflow (Recommended)

From project root:

```powershell
.\run_all_pipeline.ps1
```

What this script does:

1. Creates outputs folders if missing
2. Checks and installs core Python packages if missing
3. Runs environment sanity check (Torch/CUDA)
4. Runs main.py to save an initial BEV visualization
5. Runs pretrained PointPillars evaluation
6. Runs pretrained SECOND evaluation
7. Saves all visualization plots through scripts/save_all_visualizations.py
8. Prints saved visuals and latest eval logs

If execution policy blocks scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all_pipeline.ps1
```

---

## 6. Manual Workflow (Step-by-Step)

### Step A: Initial sanity + BEV save

```powershell
python main.py
```

Output example:

- outputs/visuals/bev_raw.png

### Step B: Classical ML pipeline

```powershell
python main_classical_ml.py
```

Typical outputs:

- outputs/extracted_dataset.npz
- outputs/xgb_model.json
- outputs/visuals/confusion_matrix.png
- outputs/visuals/pr_curve.png
- outputs/visuals/feature_importance.png

### Step C: Pretrained PointPillars evaluation

```powershell
python -m deep_learning.pretrained.pointpillers.evaluate_pretrained
```

Outputs saved in OpenPCDet output tree, for example:

- custom_openpcdet/output/pointpillar/exp_B_pretrained/eval/epoch_7728/val/default

### Step D: Pretrained SECOND evaluation

```powershell
python -m deep_learning.pretrained.second.evaluate_pretrained
```

Outputs saved in OpenPCDet output tree under tag exp_C_second_pretrained.

### Step E: Save all metric visualizations

```powershell
python scripts/save_all_visualizations.py
```

Saves:

- outputs/visuals/pointpillar_pretrained_ap.png
- outputs/visuals/pointpillar_pretrained_recall.png
- outputs/visuals/second_ap.png
- outputs/visuals/second_aos.png
- outputs/visuals/second_recall.png
- outputs/visuals/pointpillar_15_ap.png
- outputs/visuals/pointpillar_15_recall.png

---

## 7. Workflow Map for New Users

Start here:

1. Prepare environment
2. Verify KITTI data under custom_openpcdet/data/kitti
3. Run run_all_pipeline.ps1
4. Check outputs/visuals for plots
5. Check custom_openpcdet/output for evaluation logs and result files

If you only need visual plots after evaluations already exist:

1. Run python scripts/save_all_visualizations.py

If you only need pretrained evaluations:

1. Run PointPillars module
2. Run SECOND module

---

## 8. Troubleshooting

### Error: ModuleNotFoundError for deep_learning

Use project root as current directory when running scripts, or run:

```powershell
cd C:\BTP\Additional\lidar-pedestrian-detection
python scripts/save_all_visualizations.py
```

The script already injects project root into sys.path.

### Error: ModuleNotFoundError for matplotlib / numpy / torch

Install dependencies:

```powershell
pip install -r requirements.txt
```

Then install matching torch and spconv as described in Section 4.

### Error: Evaluation dataset is empty (0 samples)

Check:

1. custom_openpcdet/data/kitti exists and has expected files
2. kitti_infos_val.pkl and related info files exist
3. ImageSets split files are present
4. Evaluation config points to correct DATA_PATH

### Error: SECOND evaluation exits with non-zero code

Check:

1. Checkpoint exists at deep_learning/pretrained/pth_files/second_7862.pth
2. Config exists at custom_openpcdet/tools/cfgs/second.yaml
3. OpenPCDet dependencies are installed in the active Python environment

### Warning: open3d not installed

Only interactive 3D visualization needs open3d. Install if needed:

```powershell
pip install open3d
```

---

## 9. Notes for Contributors

- Keep all generated artifacts in outputs or custom_openpcdet/output.
- Use .gitignore defaults to avoid committing datasets, checkpoints, and large binaries.
- Prefer editing deep_learning wrappers for experiment automation instead of modifying OpenPCDet internals unless required.

---

## 10. Quick Command Reference

```powershell
# Full pipeline
.\run_all_pipeline.ps1

# Classical ML only
python main_classical_ml.py

# PointPillars pretrained eval
python -m deep_learning.pretrained.pointpillers.evaluate_pretrained

# SECOND pretrained eval
python -m deep_learning.pretrained.second.evaluate_pretrained

# Save all visualization images
python scripts/save_all_visualizations.py
```

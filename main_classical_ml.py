import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split

# Pipeline imports
from classical_pipeline.dataset import build_dataset
from classical_pipeline.model import get_model, train_or_load
from classical_pipeline.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_feature_importance
)

# Config (adjust if needed)
from deep_learning.config import VELODYNE_DIR, LABEL_DIR, CALIB_DIR


def main():

    print("\n========== CLASSICAL ML PIPELINE ==========\n")

    # -------------------------------
    # Step 1: Load / Build Dataset
    # -------------------------------
    sample_files = sorted(glob.glob(os.path.join(VELODYNE_DIR, "*.bin")))

    # ===== MODE SWITCH =====
    FAST_MODE = False  # Set to True for quick testing, False for full dataset

    if FAST_MODE:
        sample_files = sample_files[::10]   # 👈 take every 10th file
        print(f"[INFO] FAST MODE: Using {len(sample_files)} samples")
    else:
        print(f"[INFO] FULL MODE: Using all {len(sample_files)} samples")

    save_path = os.path.join("outputs", "extracted_dataset.npz")

    X, y = build_dataset(
        sample_files=sample_files,
        label_dir=LABEL_DIR,
        calib_dir=CALIB_DIR,
        save_path=save_path
    )

    print(f"[INFO] Dataset Loaded: X={X.shape}, y={y.shape}")

    # -------------------------------
    # Step 2: Train-Test Split
    # -------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train: {X_tr.shape}, Test: {X_te.shape}")

    # -------------------------------
    # Step 3: Handle Class Imbalance
    # -------------------------------
    pos = np.sum(y_tr == 1)
    neg = np.sum(y_tr == 0)

    spw = (neg / pos) if pos > 0 else 1.0
    print(f"[INFO] scale_pos_weight = {spw:.2f}")

    # -------------------------------
    # Step 4: Model
    # -------------------------------
    model = get_model(spw=spw)

    model_path = os.path.join("outputs", "xgb_model.json")

    model = train_or_load(
        model,
        X_tr, y_tr,
        X_te, y_te,
        path=model_path
    )

    # -------------------------------
    # Step 5: Evaluation
    # -------------------------------
    y_prob, y_pred = evaluate_model(
        model,
        X_te,
        y_te,
        threshold=0.9   # tweak if needed
    )

    # -------------------------------
    # Step 6: Visualizations
    # -------------------------------
    print("\n[INFO] Generating plots...")

    plot_confusion_matrix(y_te, y_pred)
    plot_pr_curve(y_te, y_prob)

    feature_names = [
        'n_points','height','width','depth','volume','density',
        'dist','z_var','refl_mean','linearity','planarity','scattering'
    ]

    plot_feature_importance(model, feature_names)

    print("\n========== PIPELINE COMPLETE ==========\n")


if __name__ == "__main__":
    main()
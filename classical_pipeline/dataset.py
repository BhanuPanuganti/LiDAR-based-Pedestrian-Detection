import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from data_utils.loader import load_velodyne, load_labels, load_calib, cam_to_lidar
from classical_pipeline.roi import roi_filter
from classical_pipeline.clustering import run_dbscan
from classical_pipeline.features import extract_features
from classical_pipeline.labeling import assign_labels


def process_file(bp, label_dir, calib_dir):
    try:
        pc = load_velodyne(bp)
        if pc is None or pc.shape[0] < 50:   # 👈 early filter
            return [], []

        pcr = roi_filter(pc)
        if pcr is None or len(pcr) < 10:
            return [], []

        cl = run_dbscan(pcr)
        if cl is None:
            return [], []

        feats = []
        for lbl in set(cl):
            if lbl == -1:
                continue

            cluster_pts = pcr[cl == lbl]
            feat = extract_features(cluster_pts)

            if (
                feat and
                0.6 < feat['height'] < 2.2 and
                0.2 < feat['width'] < 1.2 and
                0.2 < feat['depth'] < 1.2 and
                feat['n_points'] > 25 and
                feat['density'] > 8 and
                feat['z_var'] < 0.5
            ):
                feats.append(feat)

        if not feats:
            return [], []

        # GT
        fid = os.path.splitext(os.path.basename(bp))[0]
        lp = os.path.join(label_dir, fid + '.txt')
        cp = os.path.join(calib_dir, fid + '.txt')

        if not (os.path.exists(lp) and os.path.exists(cp)):
            return [], []

        Tr, R0 = load_calib(cp)
        df_l = load_labels(lp)

        gts = []
        for _, row in df_l[df_l['type'] == 'Pedestrian'].iterrows():
            c = cam_to_lidar([row['x'], row['y'], row['z']], Tr, R0)
            h,w,l = row['height'], row['width'], row['length']

            gts.append({
                'x_min': c[0]-l/2, 'x_max': c[0]+l/2,
                'y_min': c[1]-w/2, 'y_max': c[1]+w/2,
                'z_min': c[2]-h/2, 'z_max': c[2]+h/2
            })

        labels = assign_labels(feats, gts)

        return feats, labels

    except Exception:
        return [], []
    

def build_dataset(sample_files, label_dir, calib_dir, save_path):

    if os.path.exists(save_path):
        print("[INFO] Loading dataset...")
        data = np.load(save_path)
        return data['X'], data['y']

    print("[INFO] Building dataset...")

    ALL_FEATS, ALL_Y = [], []

    print("[INFO] Processing files in parallel...")

    results = Parallel(n_jobs=-1)(
        delayed(process_file)(bp, label_dir, calib_dir)
        for bp in tqdm(sample_files)
    )

    for feats, labels in results:
        ALL_FEATS.extend(feats)
        ALL_Y.extend(labels)

        # Safety check (important)
        if len(ALL_FEATS) == 0:
            raise ValueError("No features extracted. Check DBSCAN / filters.")

        # Feature columns
        FEAT_COLS = ['n_points','height','width','depth','volume','density',
                    'dist','z_var','refl_mean','linearity','planarity','scattering']

        # CREATE X and y
        X = pd.DataFrame(ALL_FEATS)[FEAT_COLS].values
        y = np.array(ALL_Y)

        # Save dataset
        np.savez(save_path, X=X, y=y)
    print("[INFO] Dataset saved")

    return X, y


if __name__ == "__main__":

    import glob
    from deep_learning.config import VELODYNE_DIR, LABEL_DIR, CALIB_DIR

    sample_files = sorted(glob.glob(os.path.join(VELODYNE_DIR, "*.bin")))

    save_path = os.path.join("outputs", "extracted_dataset.npz")

    X, y = build_dataset(
        sample_files=sample_files,
        label_dir=LABEL_DIR,
        calib_dir=CALIB_DIR,
        save_path=save_path
    )

    print(f"[DONE] Dataset shape: {X.shape}, Labels: {y.shape}")
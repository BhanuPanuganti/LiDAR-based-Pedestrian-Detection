import glob
import os
from tqdm import tqdm

from config import *
from setup_env import setup_environment, check_torch
from data_utils.loader import load_velodyne, load_labels
from visualization.bev import plot_bev

def main():
    setup_environment()
    check_torch()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)

    sample_files = sorted(glob.glob(os.path.join(VELODYNE_DIR, '*.bin')))

    print(f"Scans found: {len(sample_files)}")

    pc0 = load_velodyne(sample_files[0])
    print(f"Sample scan: {pc0.shape[0]:,} points")

    # Load labels
    label_files = glob.glob(os.path.join(LABEL_DIR,'*.txt'))

    all_lbl = []
    for p in tqdm(label_files, desc="labels"):
        df = load_labels(p)
        all_lbl.append(df)

    import pandas as pd
    all_lbl = pd.concat(all_lbl, ignore_index=True)

    print("\nClass counts:")
    print(all_lbl['type'].value_counts())

    plot_bev(
        pc0,
        title=f'Raw BEV — {pc0.shape[0]:,} points',
        save_path=os.path.join(VISUALS_DIR, 'bev_raw.png')
    )

if __name__ == "__main__":
    main()
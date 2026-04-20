import numpy as np
import pandas as pd
import os

def load_velodyne(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    except Exception as e:
        print(f"[ERROR] load_velodyne: {e}")
        return None


def load_labels(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 15:
                    continue

                rows.append({
                    'type': p[0],
                    'truncated': float(p[1]),
                    'occluded': int(p[2]),
                    'height': float(p[8]),
                    'width': float(p[9]),
                    'length': float(p[10]),
                    'x': float(p[11]),
                    'y': float(p[12]),
                    'z': float(p[13]),
                    'rotation_y': float(p[14])
                })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"[ERROR] load_labels: {e}")
        return pd.DataFrame()


def load_calib(path):
    try:
        data = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    data[k.strip()] = np.array([float(x) for x in v.split()])

        return data['Tr_velo_to_cam'].reshape(3,4), data['R0_rect'].reshape(3,3)

    except Exception as e:
        print(f"[ERROR] load_calib: {e}")
        return None, None


def cam_to_lidar(xyz, Tr, R0):
    try:
        pt = np.array([*xyz, 1.0])
        Tr4 = np.vstack([Tr,[0,0,0,1]])
        R04 = np.eye(4)
        R04[:3,:3] = R0

        return (np.linalg.inv(R04 @ Tr4) @ pt)[:3]

    except Exception as e:
        print(f"[ERROR] cam_to_lidar: {e}")
        return None
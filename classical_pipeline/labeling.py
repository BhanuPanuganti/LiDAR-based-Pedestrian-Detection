import numpy as np

IOU_THR = 0.10

def iou3d(b1, b2):
    ix = max(0, min(b1['x_max'], b2['x_max']) - max(b1['x_min'], b2['x_min']))
    iy = max(0, min(b1['y_max'], b2['y_max']) - max(b1['y_min'], b2['y_min']))
    iz = max(0, min(b1['z_max'], b2['z_max']) - max(b1['z_min'], b2['z_min']))

    inter = ix * iy * iz

    v1 = max((b1['x_max']-b1['x_min']) *
             (b1['y_max']-b1['y_min']) *
             (b1['z_max']-b1['z_min']), 1e-6)

    v2 = max((b2['x_max']-b2['x_min']) *
             (b2['y_max']-b2['y_min']) *
             (b2['z_max']-b2['z_min']), 1e-6)

    return inter / (v1 + v2 - inter + 1e-6)


def assign_labels(features, gts):
    try:
        labels = []

        for f in features:
            pred = {
                'x_min': f['_xmin'], 'x_max': f['_xmax'],
                'y_min': f['_ymin'], 'y_max': f['_ymax'],
                'z_min': f['_zmin'], 'z_max': f['_zmax']
            }

            iou_max = max((iou3d(pred, g) for g in gts), default=0)

            labels.append(1 if iou_max >= IOU_THR else 0)

        return np.array(labels)

    except Exception as e:
        print(f"[ERROR] labeling failed: {e}")
        return None
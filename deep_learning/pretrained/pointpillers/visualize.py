import matplotlib.pyplot as plt
import numpy as np

def plot_pretrained_ap(save_path=None):

    categories = ['Easy','Moderate','Hard']

    bbox = [66.06, 61.25, 58.53]
    bev  = [64.07, 59.88, 56.13]
    ap3d = [57.48, 53.28, 49.49]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x-width, bbox, width, label="BBox")
    plt.bar(x, bev, width, label="BEV")
    plt.bar(x+width, ap3d, width, label="3D")

    plt.xticks(x, categories)
    plt.ylabel("AP %")
    plt.title("Pretrained PointPillars (100%)")
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_pretrained_recall():
    import matplotlib.pyplot as plt

    iou = [0.3, 0.5, 0.7]
    recall = [94.33, 89.31, 67.59]

    plt.plot(iou, recall, marker='o')
    plt.title("Pretrained Recall")
    plt.xlabel("IoU")
    plt.ylabel("Recall (%)")
    plt.show()
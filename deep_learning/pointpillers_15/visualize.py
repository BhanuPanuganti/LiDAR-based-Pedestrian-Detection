import matplotlib.pyplot as plt
import os

def plot_loss(steps, losses, save_path):
    plt.figure(figsize=(10,4))
    plt.plot(steps, losses, color='orange')
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.show()


def plot_ap_metrics(save_path):
    import numpy as np

    categories = ['Easy','Moderate','Hard']

    bbox = [59.36, 55.05, 52.29]
    bev  = [56.00, 53.05, 48.93]
    ap3d = [51.04, 47.51, 43.67]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x-width, bbox, width, label="BBox")
    plt.bar(x, bev, width, label="BEV")
    plt.bar(x+width, ap3d, width, label="3D")

    plt.xticks(x, categories)
    plt.ylabel("AP %")
    plt.title("PointPillars (15%) Performance")
    plt.legend()

    plt.savefig(save_path)
    plt.show()

def plot_recall(save_path):
    import matplotlib.pyplot as plt

    iou = [0.3, 0.5, 0.7]
    recall = [95.46, 90.34, 71.38]

    plt.plot(iou, recall, marker='o')
    plt.xlabel("IoU")
    plt.ylabel("Recall (%)")
    plt.title("Recall vs IoU")

    plt.savefig(save_path)
    plt.show()
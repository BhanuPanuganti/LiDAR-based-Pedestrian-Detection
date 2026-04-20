import matplotlib.pyplot as plt
import numpy as np

def plot_second_ap():

    categories = ['Easy','Moderate','Hard']

    bbox = [72.39, 68.69, 65.99]
    bev  = [67.49, 64.83, 61.45]
    ap3d = [60.67, 57.80, 54.15]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10,6))

    plt.bar(x-width, bbox, width, label="BBox")
    plt.bar(x, bev, width, label="BEV")
    plt.bar(x+width, ap3d, width, label="3D")

    plt.xticks(x, categories)
    plt.title("SECOND Model Performance")
    plt.ylabel("AP %")
    plt.legend()
    plt.show()

def plot_second_aos():

    labels = ['Easy','Moderate','Hard']
    aos = [68.04, 63.91, 60.70]

    plt.bar(labels, aos)
    plt.title("SECOND AOS")
    plt.ylabel("AOS %")
    plt.show()

def plot_second_recall():

    iou = [0.3, 0.5, 0.7]
    recall = [95.46, 90.34, 71.38]

    plt.plot(iou, recall, marker='o')
    plt.title("SECOND Recall")
    plt.xlabel("IoU")
    plt.ylabel("Recall (%)")
    plt.show()


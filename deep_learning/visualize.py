import os

import matplotlib.pyplot as plt


def plot_loss(steps, losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, color='orange')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

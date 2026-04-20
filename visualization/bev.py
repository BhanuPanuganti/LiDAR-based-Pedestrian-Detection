import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_bev(pc, boxes=None, pred_boxes=None, title='BEV', save_path=None):
    try:
        fig, ax = plt.subplots(figsize=(14,6))

        mask = (pc[:,0]>0)&(pc[:,0]<50)&(pc[:,1]>-15)&(pc[:,1]<15)
        r = pc[mask]

        ax.scatter(r[:,0], r[:,1], s=0.3, c=r[:,2],
                   cmap='viridis', vmin=-2, vmax=2, alpha=0.5)

        for bset, col, ls in [(boxes,'lime','-'), (pred_boxes,'red','--')]:
            if bset:
                for b in bset:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (b['x_min'], b['y_min']),
                            b['x_max'] - b['x_min'],
                            b['y_max'] - b['y_min'],
                            linewidth=2,
                            edgecolor=col,
                            facecolor='none',
                            linestyle=ls
                        )
                    )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show()

    except Exception as e:
        print(f"[ERROR] plot_bev: {e}")
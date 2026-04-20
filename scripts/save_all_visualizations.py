import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_learning.pretrained.pointpillers.visualize import (
    plot_pretrained_ap,
    plot_pretrained_recall,
)
from deep_learning.pretrained.second.visualize_metrics import (
    plot_second_ap,
    plot_second_aos,
    plot_second_recall,
)
from deep_learning.pointpillers_15.visualize import (
    plot_ap_metrics,
    plot_recall,
)


def main():
    out_dir = os.path.join('outputs', 'visuals')
    os.makedirs(out_dir, exist_ok=True)

    plot_pretrained_ap(os.path.join(out_dir, 'pointpillar_pretrained_ap.png'))
    plt.close('all')

    plot_pretrained_recall()
    plt.savefig(os.path.join(out_dir, 'pointpillar_pretrained_recall.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    plot_second_ap()
    plt.savefig(os.path.join(out_dir, 'second_ap.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    plot_second_aos()
    plt.savefig(os.path.join(out_dir, 'second_aos.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    plot_second_recall()
    plt.savefig(os.path.join(out_dir, 'second_recall.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    plot_ap_metrics(os.path.join(out_dir, 'pointpillar_15_ap.png'))
    plt.close('all')

    plot_recall(os.path.join(out_dir, 'pointpillar_15_recall.png'))
    plt.close('all')

    print('[INFO] Visualization files saved in outputs/visuals')


if __name__ == '__main__':
    main()

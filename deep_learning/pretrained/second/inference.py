import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CUSTOM_ROOT = os.path.join(BASE_DIR, 'custom_openpcdet')

sys.path.insert(0, CUSTOM_ROOT)
sys.path.insert(0, os.path.join(CUSTOM_ROOT, 'tools'))

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def get_single_prediction(bin_path, cfg_file, ckpt_path):

    logger = common_utils.create_logger()

    cfg_from_yaml_file(cfg_file, cfg)

    class SingleFrameDataset(DatasetTemplate):
        def __len__(self): return 1

        def __getitem__(self, index):
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            return self.prepare_data({'points': points, 'frame_id': '0'})

    ds = SingleFrameDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        logger=logger
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=ds
    )

    model.load_params_from_file(
        filename=ckpt_path,
        logger=logger,
        to_cpu=True
    )

    model.cuda().eval()

    with torch.no_grad():
        batch = ds.collate_batch([ds[0]])
        load_data_to_gpu(batch)
        pred_dicts, _ = model.forward(batch)

    return pred_dicts[0]
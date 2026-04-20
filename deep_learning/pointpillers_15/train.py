import os
from deep_learning.config import TOOLS_DIR, CONFIG_FILE, EXP_TAG

def print_train_command():
    print(f"""
Run this in terminal:

cd {TOOLS_DIR}

python train.py ^
  --cfg_file {CONFIG_FILE} ^
  --batch_size 2 ^
  --epochs 25 ^
  --extra_tag {EXP_TAG}
""")
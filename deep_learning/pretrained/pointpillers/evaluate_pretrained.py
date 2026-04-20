import os
import subprocess
import sys

from deep_learning.config import *

# define required paths
TOOLS_DIR = os.path.join(OPENPCDET_ROOT, "tools")
CONFIG_FILE = os.path.join(OPENPCDET_ROOT, "tools", "cfgs", "pointpillar.yaml")

def run_evaluation():

    print("\n========== RUNNING EVALUATION ==========\n")

    ckpt_path = POINTPILLAR_PRETRAINED_CKPT

    # Debug prints
    print("TOOLS_DIR:", TOOLS_DIR)
    print("CONFIG_FILE:", CONFIG_FILE)
    print("CKPT_PATH:", ckpt_path)
    print("CKPT EXISTS:", os.path.exists(ckpt_path))

    if not os.path.exists(ckpt_path):
        print("\nCheckpoint not found. Fix path and retry.")
        return

    print("\n[INFO] Running OpenPCDet evaluation...\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "test.py",
                "--cfg_file", CONFIG_FILE,
                "--batch_size", "4",
                "--ckpt", ckpt_path,
                "--extra_tag", "exp_B_pretrained"
            ],
            cwd=TOOLS_DIR
        )

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")


if __name__ == "__main__":
    run_evaluation()
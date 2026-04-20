import cmd
import os

import sys, os
import subprocess


from deep_learning.config import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUSTOM_ROOT = os.path.join(BASE_DIR, "custom_openpcdet")

sys.path.insert(0, CUSTOM_ROOT)
sys.path.insert(0, os.path.join(CUSTOM_ROOT, "tools"))


import subprocess

def run_inference(tools_dir, config_file, ckpt_path, sample_path):
    try:
        print("[INFO] Running inference...")
        print("TOOLS DIR:", tools_dir)

        subprocess.run(
            [
                "python",
                "demo.py",
                "--cfg_file", config_file,
                "--ckpt", ckpt_path,
                "--data_path", sample_path
            ],
            cwd=tools_dir
        )

    except Exception as e:
        print(f"[ERROR] inference failed: {e}")

def main():

    tools_dir = os.path.join(CUSTOM_ROOT, "tools")

    config_file = POINTPILLAR_PED_CFG
    ckpt_path   = POINTPILLAR_15_CKPT
    sample_path = SAMPLE_FILE

    run_inference(tools_dir, config_file, ckpt_path, sample_path)


if __name__ == "__main__":
    main()
import os

def run_evaluation(tools_dir, config_file, ckpt_path, tag):
    try:
        cmd = f"cd {tools_dir} && python test.py --cfg_file {config_file} --batch_size 2 --workers 2 --ckpt {ckpt_path} --extra_tag {tag}"

        print("[INFO] Running evaluation...")
        os.system(cmd)

    except Exception as e:
        print(f"[ERROR] evaluation failed: {e}")
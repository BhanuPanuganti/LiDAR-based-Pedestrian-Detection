import os
import subprocess
import sys

from deep_learning.config import OPENPCDET_ROOT, SECOND_PRETRAINED_CKPT


TOOLS_DIR = os.path.join(OPENPCDET_ROOT, 'tools')
CONFIG_FILE = os.path.join(TOOLS_DIR, 'cfgs', 'second.yaml')
EXP_TAG = 'exp_C_second_pretrained'


def run_evaluation():
  print('\n========== RUNNING SECOND EVALUATION ==========\n')

  print('TOOLS_DIR:', TOOLS_DIR)
  print('CONFIG_FILE:', CONFIG_FILE)
  print('CKPT_PATH:', SECOND_PRETRAINED_CKPT)
  print('CKPT EXISTS:', os.path.exists(SECOND_PRETRAINED_CKPT))

  if not os.path.exists(SECOND_PRETRAINED_CKPT):
    print('\nCheckpoint not found. Fix path and retry.')
    return

  if not os.path.exists(CONFIG_FILE):
    print('\nConfig file not found. Fix path and retry.')
    return

  print('\n[INFO] Running OpenPCDet SECOND evaluation...\n')

  try:
    subprocess.run(
      [
        sys.executable,
        'test.py',
        '--cfg_file', CONFIG_FILE,
        '--batch_size', '4',
        '--workers', '2',
        '--ckpt', SECOND_PRETRAINED_CKPT,
        '--extra_tag', EXP_TAG,
      ],
      cwd=TOOLS_DIR,
      check=True,
    )
  except subprocess.CalledProcessError as exc:
    print(f'\n[ERROR] SECOND evaluation failed with exit code {exc.returncode}.')


if __name__ == '__main__':
  run_evaluation()
import glob
import os
import re


def parse_training_logs(directory):
    steps, losses = [], []

    log_files = glob.glob(os.path.join(directory, 'train_*.log'))
    log_files.sort(key=os.path.getmtime)

    for path in log_files:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(r'loss:\s*([\d.]+)', line.lower())
                if match:
                    losses.append(float(match.group(1)))
                    steps.append(len(losses))

    return steps, losses

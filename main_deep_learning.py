from deep_learning.config import *
from deep_learning.logs import parse_training_logs
from deep_learning.visualize import plot_loss

def main():

    steps, losses = parse_training_logs(EXP_DIR)

    if losses:
        plot_loss(steps, losses, "visuals/loss.png")
    else:
        print("No logs found")

if __name__ == "__main__":
    main()
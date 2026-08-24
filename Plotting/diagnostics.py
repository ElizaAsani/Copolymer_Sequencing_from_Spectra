"""
Diagnostic plotting functions for training and validating model.
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(output_dir):
    """
    Plot train/validation loss curves from loss_history.csv stored in output_dir.
    """

    loss_df = pd.read_csv(os.path.join(output_dir, "loss_history.csv"))

    _, ax = plt.subplots()
    ax.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')
    ax.plot(loss_df['epoch'], loss_df['validate_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cross Entropy Loss')
    ax.legend()

    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    return

PLOTS = {
    "loss": plot_loss
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot diagnostic plots from training/validation.")
    parser.add_argument("output_dir", type=str, help="Path to a run's output directory")
    parser.add_argument("--only", type=str, choices=list(PLOTS), default=None, help="Type of plot to generate (default: all)")

    args = parser.parse_args()

    plots_to_run = [args.only] if args.only else list(PLOTS)
    for name in plots_to_run:
        PLOTS[name](args.output_dir)
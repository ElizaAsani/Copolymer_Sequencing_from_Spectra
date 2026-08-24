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

    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    return

def plot_errors(output_dir):
    """Function to visualize number of errors in generated sequences.
    """ 

    error_df = pd.read_csv(os.path.join(output_dir, "errors", "errors.csv"))
    
    # plot count of number of errors 
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    error_idxs = error_df['num_errors'].value_counts().sort_index()
    axs[0].bar(error_idxs.index, error_idxs.values)
    axs[0].set_title("Count of Number of Errors")
    axs[0].set_xlabel("Number of Errors")
    axs[0].set_ylabel("Count")

    # plot avg number of errors vs. sequence length
    avg_error = error_df[['sequence_length', 'num_errors']].groupby('sequence_length').mean()
    axs[1].bar(avg_error.index, avg_error['num_errors'])
    axs[1].set_title("Avg. Number of Errors vs. Sequence Length")
    axs[1].set_xlabel("Sequence Length")
    axs[1].set_ylabel("Average Number of Errors")

    plt.savefig(os.path.join(output_dir, "errors", "errors.png"))
    plt.close()

    return

PLOTS = {
    "loss": plot_loss,
    "errors": plot_errors
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot diagnostic plots from training/validation.")
    parser.add_argument("output_dir", type=str, help="Path to a run's output directory")
    parser.add_argument("--only", type=str, choices=list(PLOTS), default=None, help="Type of plot to generate (default: all)")

    args = parser.parse_args()

    plots_to_run = [args.only] if args.only else list(PLOTS)
    for name in plots_to_run:
        PLOTS[name](args.output_dir)
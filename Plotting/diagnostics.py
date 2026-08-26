"""
Diagnostic plotting functions for training and validating model.
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

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
    """Function to visualize number of errors in predicted sequences.
    """ 

    error_df = pd.read_csv(os.path.join(output_dir, "errors", "errors.csv"))
    
    # plot histogram of number of errors 
    _, axs = plt.subplots(1, 2, figsize=(12, 4))
    error_idxs = error_df['num_errors'].value_counts().sort_index()
    axs[0].bar(error_idxs.index, error_idxs.values)
    axs[0].set_title("Histogram of Number of Prediction Errors")
    axs[0].set_xlabel("Number of Prediction Errors")
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

def plot_beam_histogram(output_dir, alpha=1, threshold=1):

    fig, axs = plt.subplots(2, 5, layout="constrained", figsize=(20,6))
    fig.suptitle("Decoding Distribution for Individual Copolymer Sequences", fontsize=16)
    style = {'facecolor': '#99CCFF', 'edgecolor': 'C0', 'linewidth': 3}
    style_2 = {'facecolor': '#9FD1AC', 'edgecolor': '#095911', 'linewidth': 3}
    
    # load dataset
    beam_search_dir = os.path.join(output_dir, "beam_search", f"alpha_{alpha}")
    decodings = pd.read_csv(os.path.join(beam_search_dir, "beam_search.csv"))
    i = 0
    # enumerate through axes on plot
    for ax in axs.flatten():
        # get a sequence whose decodings are not certain
        continue_search = True
        while continue_search:
            score = ast.literal_eval(decodings['Scores'][i])
            if max(score) < threshold:
                continue_search = False
            else:
                i += 1
        continue_search = True
        
        # get decodings for single sequence
        seq = decodings['Target Sequence'][i]   
        predictions = decodings['Predicted Sequences'][i]
        scores = decodings['Scores'][i]
        num_errors = decodings['Num Errors'][i] 

        # convert predictions to array of strings
        predictions = ast.literal_eval(predictions)
        scores = np.round(ast.literal_eval(scores), 4)
        num_errors = ast.literal_eval(num_errors)

        # create dataframe of predictions, scores, and errors
        scores = {'Predictions': predictions, 'Scores': scores, 'Errors': num_errors}
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values(by='Errors')
        
        # generate text box of predictions
        textBox = "Predictions: \n"
        for pred_seq, score in zip(scores_df['Predictions'], scores_df['Scores']):
            if pred_seq == seq:
                textBox += f"*{pred_seq}: {score} \n"
            else:
                textBox += f"{pred_seq}: {score} \n"

        # plot histogram as bar chart
        container = ax.bar(scores_df['Predictions'], scores_df['Scores'], **style)
        # color the correct bar 
        try:
            correct_idx = scores_df['Predictions'].tolist().index(seq)
            container.patches[correct_idx].set(**style_2, label='correct')
        except ValueError:
            pass
        try:
            correct_idx = scores_df['Predictions'].tolist().index(seq[::-1]) # try reverse match
            container.patches[correct_idx].set(**style_2, label='correct')
        except ValueError:
            pass
        ax.set_title(f'Expected: {seq}')
        ax.set_ylabel('Score')       
        ax.set_xlabel('Number of Errors') 
        # set ticks to be the number of errors
        ax.set_xticks(np.arange(0, len(scores_df)), labels=scores_df['Errors'])
        ax.set_ylim(0, 1)
        ax.text(0.95, 0.95, textBox, transform=ax.transAxes, fontsize=8, horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))        
    
        i += 1

    plt.savefig(os.path.join(beam_search_dir, f"beam_histogram_thresh_{threshold}.png"))
    plt.close()

    return

PLOTS = {
    "loss": plot_loss,
    "errors": plot_errors,
    "beam": plot_beam_histogram
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot diagnostic plots from training/validation.")
    parser.add_argument("output_dir", type=str, help="Path to a run's output directory")
    parser.add_argument("--only", type=str, choices=list(PLOTS), default=None, help="Type of plot to generate (default: all)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for beam search plot (default: 1.0)")
    parser.add_argument("--th", type=float, default=1, help="Threshold value for beam search plot (default: 1)")

    args = parser.parse_args()

    plots_to_run = [args.only] if args.only else list(PLOTS)
    for name in plots_to_run:
        if name == "beam":
            PLOTS[name](args.output_dir, alpha=args.alpha, threshold=args.th)
        else:
            PLOTS[name](args.output_dir)

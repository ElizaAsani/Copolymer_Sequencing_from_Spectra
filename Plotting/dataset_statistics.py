"""
Plots to visualize the sequence distributions in the dataset.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Multispectra-to-sequence Transformer'))
from SequenceEncoder import stratified_split_indices

palette = ['#6A4E5A','#A72420', '#909033', '#E0884D']

colors = {
    "dataset": palette[0],          
    "train": palette[1],            
    "validate": palette[2],                  
    "test": palette[3],                  
}

def load_sequences(sequence_file):
    """Load sequences from a CSV file."""
    df = pd.read_csv(sequence_file, header=None, names=['Sequence', 'Lambda'])

    train_idx, val_idx, test_idx = stratified_split_indices([len(seq) for seq in df['Sequence']]) # same as model
    splits = {'dataset': df, 'train': df.iloc[train_idx], 'validate': df.iloc[val_idx], 'test': df.iloc[test_idx]}
    
    return splits

def plot_sequence_length_distribution(splits):
    """Plot the distribution of sequence lengths for each split."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [1, 2]})

    split_names = ['train', 'validate', 'test']
    min_length = min(len(seq) for seq in splits['dataset']['Sequence'])
    max_length = max(len(seq) for seq in splits['dataset']['Sequence'])

    sequence_lengths = np.arange(min_length, max_length + 1)
    bottom = np.zeros(len(sequence_lengths))
    break_point = 6  # Split the x-axis into two parts for better visualization

    for split_name in split_names:
        counts = np.bincount(
            [len(seq) for seq in splits[split_name]['Sequence']],
            minlength=max_length + 1,
        )[min_length:max_length + 1]
        axs[0].bar(
            sequence_lengths[:break_point],
            counts[:break_point],
            bottom=bottom[:break_point],
            width=0.8,
            label=split_name.capitalize(),
            color=colors[split_name]
        )
        axs[1].bar(
            sequence_lengths[break_point:],
            counts[break_point:],
            bottom=bottom[break_point:],
            width=0.8,
            label=split_name.capitalize(),
            color=colors[split_name],
        )
        bottom += counts

    axs[0].set_xticks(sequence_lengths[:break_point])
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Count')
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    axs[1].set_xticks(sequence_lengths[break_point:])
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('Count')
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('sequence_length_distribution.svg')

def plot_sequence_correlation_distribution(splits):
    """Plot the distribution of sequence correlations for each split."""
    fig, ax = plt.subplots(figsize=(5, 3))

    split_names = ['train', 'validate', 'test']

    for split_name in split_names:
        sns.kdeplot(splits[split_name]['Lambda'], 
                    label=split_name.capitalize(), 
                    color=colors[split_name], 
                    fill=True, 
                    alpha=0.25, 
                    ax=ax)

    ax.set_xlim(-1, 1)
    ax.set_xlabel('Sequence Correlation')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('sequence_correlation_distribution.svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot statistics for a given dataset.")
    parser.add_argument("--sequence_file", type=str, required=True, help="Path to the sequence file, relative to ../Data Generation/.")
    args = parser.parse_args()

    sequence_file = os.path.join(os.path.dirname(__file__), '..', 'Data Generation', args.sequence_file)
    splits = load_sequences(sequence_file)
    plot_sequence_length_distribution(splits)
    plot_sequence_correlation_distribution(splits)
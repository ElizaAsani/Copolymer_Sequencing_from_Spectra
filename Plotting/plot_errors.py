import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Multispectra-to-sequence Transformer'))
from SequenceEncoder import stratified_split_indices

def plot_errors(folders, colors, sequence_file):

    error_dfs = read_errors(folders, sequence_file)
    out_folder = os.path.join(folders['UV-Vis + MS + NMR'], "errors")

    plot_avg_errors(error_dfs, colors, out_folder)
    handles, labels = plot_reconstruction_length(error_dfs, colors, out_folder)
    plot_reconstruction_lambda(error_dfs, colors, out_folder)

    plot_legend(handles, labels)

    return

def read_errors(folders, sequence_file):

    # get lambda for each sequence in the test set
    sequence_df = pd.read_csv(sequence_file, header=None, names=['Sequence', 'Lambda'])
    _, _, test_idx = stratified_split_indices([len(seq) for seq in sequence_df['Sequence']])
    test_df = sequence_df.iloc[test_idx]
    lambdas = dict(zip(test_df['Sequence'], test_df['Lambda']))

    error_dfs = {}
    for spectrum, folder in folders.items():
        error_df = pd.read_csv(os.path.join(folder, "errors", "errors.csv"))
        error_df['reconstructed'] = error_df['num_errors'] == 0
        error_df['lambda'] = error_df['target_sequence'].map(lambdas)
        error_dfs[spectrum] = error_df

    return error_dfs

def plot_avg_errors(error_dfs, colors, out_folder):

    _, ax = plt.subplots(figsize=(2, 2), layout='constrained')
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    for spectrum, error_df in error_dfs.items():
        avg_errors = error_df.groupby('sequence_length')['num_errors'].mean()
        ax.plot(avg_errors.index, avg_errors.values, marker='o', markerfacecolor='white', markersize=3, 
                color=colors[spectrum], label=spectrum)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Avg. Number of Errors')
    ax.set_xticks(np.arange(avg_errors.index.min(), avg_errors.index.max() + 1, step=3))

    plt.savefig(os.path.join(out_folder, "errors_avg_vs_length.svg"))
    plt.close()

    return

def plot_reconstruction_length(error_dfs, colors, out_folder):

    _, ax = plt.subplots(figsize=(2, 2), layout='constrained')
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    for spectrum, error_df in error_dfs.items():
        reconstruction = error_df.groupby('sequence_length')['reconstructed'].mean() * 100
        ax.plot(reconstruction.index, reconstruction.values, marker='o', markerfacecolor='white', markersize=3, 
                color=colors[spectrum], label=spectrum)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('% Sequence Reconstruction')
    ax.set_xticks(np.arange(reconstruction.index.min(), reconstruction.index.max() + 1, step=3))
    ax.set_ylim(0, 105)
    handles, labels = ax.get_legend_handles_labels()

    plt.savefig(os.path.join(out_folder, "errors_recon_vs_length.svg"))
    plt.close()

    return handles, labels

def plot_reconstruction_lambda(error_dfs, colors, out_folder):

    edges = np.linspace(-1, 1, 9)
    centers = (edges[:-1] + edges[1:]) / 2

    _, ax = plt.subplots(figsize=(2, 2), layout='constrained')
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    for spectrum, error_df in error_dfs.items():
        bins = pd.cut(error_df['lambda'], bins=edges, include_lowest=True, labels=centers)
        reconstruction = error_df.groupby(bins, observed=True)['reconstructed'].mean() * 100
        ax.plot(reconstruction.index.astype(float), reconstruction.values, marker='o', markerfacecolor='white', 
                markersize=3, color=colors[spectrum], label=spectrum)
    ax.set_xlabel(r'Sequence Correlation ($\lambda$)')
    ax.set_ylabel('% Sequence Reconstruction')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, 105)

    plt.savefig(os.path.join(out_folder, "errors_recon_vs_lambda.svg"))
    plt.close()

    return

def plot_legend(handles, labels):
    _, ax = plt.subplots(figsize=(2, 0.5))
    ax.legend(handles=handles, labels=labels, loc='center', frameon=False, ncol=len(labels))
    ax.axis('off')
    plt.savefig(f'./errors_legend.svg')
    plt.close()

    return
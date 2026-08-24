import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_degeneracies(degeneracy_folders, colors):

    fig = plt.figure(figsize=(3.25, 2), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.16])

    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    legend_ax = fig.add_subplot(gs[1, :])

    for spectrum, folder in degeneracy_folders.items():
        degen_file = os.path.join(folder, "degeneracy.csv")
        degen_df = pd.read_csv(degen_file)
        seq_lengths = degen_df['Sequence Lengths']
        total = degen_df['Total # Sequences']
        unique = degen_df['Total # Unique Spectra']
        percents = [round(num_unique / num_total, 2)*100 for num_unique, num_total in zip(unique, total)]

        axs[0].plot(seq_lengths, percents, marker='o', markerfacecolor='white', markersize=3, color=colors[spectrum], label=spectrum)
        
        largest = degen_df['Largest Group']
        axs[1].plot(seq_lengths, largest, marker='o', markerfacecolor='white', markersize=3, color=colors[spectrum], label=spectrum)
    
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('% Unique Spectra')
    axs[0].set_xticks(np.arange(min(seq_lengths), max(seq_lengths) + 1, step=3))

    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('Largest Spectral\nDegeneracy')
    axs[1].set_yscale('log')
    axs[1].set_xticks(np.arange(min(seq_lengths), max(seq_lengths) + 1, step=3))
    
    handles, labels = axs[1].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', frameon=False, ncol=3)
    legend_ax.axis('off')

    plt.savefig('degeneracies.svg')
    plt.close()

    return 
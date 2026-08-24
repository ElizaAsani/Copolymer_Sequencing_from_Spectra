import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from subgroup_vector_distance import subgroup_vector_distance

def plot_subgroup_distances(folders, colors, subgroup_type='NMR', distance_type='manhattan'):

    # get distance counts
    distance_counts = {}
    for spectrum, folder in folders.items():
        distance_counts[spectrum] = plot_subgroup_distance(folder, colors[spectrum], subgroup_type, distance_type)

    # get counts for distance zero
    total = sum(distance_counts['NMR'].values)
    zero_distance_counts = {spectrum: distance_count.get(0) / total * 100 for spectrum, distance_count in distance_counts.items()}

    # get largest distance
    max_distance = {spectrum: distance_count.index[-1] for spectrum, distance_count in distance_counts.items()}

    fig = plt.figure(figsize=(4, 2.25), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.16])

    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    legend_ax = fig.add_subplot(gs[1, :])

    axs[0].grid(axis='y', alpha=0.7)
    axs[0].set_axisbelow(True)
    axs[0].bar(zero_distance_counts.keys(), zero_distance_counts.values(), color=colors.values(), width=0.6)
    axs[0].tick_params(axis='x', labelbottom=False)
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel ('% Exact Subgroup \n Recovery')                  #(r"% $||f_{{pred}} - f_{{target}}||_1 = 0 $")

    axs[1].grid(axis='y', alpha=0.7)
    axs[1].set_axisbelow(True)
    axs[1].bar(max_distance.keys(), max_distance.values(), color=colors.values(), width=0.6)
    axs[1].tick_params(axis='x', labelbottom=False)
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel("Max Subgroup Error")         #(r'$max(||f_{{pred}} - f_{{target}}||_1) $')
    axs[1].set_ylim(0, 30)

    handles = [Patch(facecolor=colors[name], edgecolor='none', label=name) for name in colors]
    legend_ax.legend(handles=handles, loc='center', frameon=False, ncol=2)
    legend_ax.axis('off')    

    plt.savefig(os.path.join(folders['UV-Vis + MS + NMR'], "errors", f"{distance_type}_distance_distributions.svg"))
    plt.close()

    plot_legend(handles, colors.keys())

    return

def plot_subgroup_distance(folder, color, subgroup_type='NMR', distance_type='manhattan'):

    out_folder = os.path.join(folder, "errors")
    error_df = pd.read_csv(os.path.join(out_folder, "errors.csv"))

    # calculate subgroup distances
    error_df['distances'] = [subgroup_vector_distance(target, prediction, subgroup_type, distance_type) 
                             for target, prediction in zip(error_df['target sequence'], error_df['predicted sequence'])]
    distance_counts = error_df['distances'].value_counts().sort_index()
    max_dist = max(distance_counts.index)

    # plot distribution of distances
    _, ax = plt.subplots(figsize=(3, 2))
    bars = ax.bar(distance_counts.index, distance_counts.values, color=color)
    ax.set_yscale('log')
    #ax.bar_label(bars, fontsize=8)
    #ax.set_title(f'{distance_type.capitalize()} Distances Between Target and Predicted Subgroup Counts')
    ax.set_xlabel(f'Subgroup Error')
    ax.set_ylabel('Frequency')
    ax.set_xticks(np.arange(max_dist + 1, step=(max_dist // 15) + 1))
    #ax.set_ylim(0, 1.5*max(distance_counts.values))
    ax.set_ylim(0, 10**4)
    plt.savefig(os.path.join(out_folder, f"{distance_type}_distance_distribution.svg"))
    plt.close()

    return distance_counts

def plot_legend(handles, labels):
    _, ax = plt.subplots(figsize=(2, 0.5))
    ax.legend(handles=handles, labels=labels, loc='center', frameon=False, ncol=len(labels))
    ax.axis('off')
    plt.savefig(f'./dist_legend.svg')
    plt.close()

    return
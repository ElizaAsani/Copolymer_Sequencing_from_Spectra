"""
Sequence reconstruction plotting script.
"""

import os

import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass 
class ModelOutput:
    greedy: float
    top1: float
    top5: float
    top10: float

## ------ Extract Expected Reconstruction ------ ##

def read_expected_reconstruction(folder):

    dataset_degeneracies = os.path.join(folder, "dataset", "dataset_degeneracies.txt")

    with open(dataset_degeneracies) as f:
        lines = [line.strip() for line in f.readlines()]

    test_idx = lines.index("Test set")
    expected_line = lines[test_idx + 5] # "Expected % Correctly Reconstructed: XX.XX%"
    return float(expected_line.split(":")[1].split("%")[0].strip())

## ------ Extract Model Output ------ ##

def read_model_output(folder, alpha):

    with open(os.path.join(folder, "output.txt"), "r") as f:
        lines = f.readlines()
        greedy = round(float(lines[0].split(":")[1].strip())*100, 2)

    with open(os.path.join(folder, "beam_search", f"alpha_{alpha}", "beam_search.txt"), "r") as f:
        lines = f.readlines()
        top1 = round(float(lines[5].split(":")[1].split("%")[0].strip()), 2)
        top5 = round(float(lines[4].split(":")[1].split("%")[0].strip()), 2)
        top10 = round(float(lines[3].split(":")[1].split("%")[0].strip()), 2)

    return ModelOutput(greedy=greedy, top1=top1, top5=top5, top10=top10)

## ------ Plot Sequence Reconstruction ------ ##

def plot_sequence_reconstruction(folders, colors, alpha, degeneracy_folders=None):

    model_outputs = {spectrum: read_model_output(folder, alpha) for spectrum, folder in folders.items()}

    algorithms = ['Greedy', 'Top1', 'Top5', 'Top10']

    fig = plt.figure(figsize=(2.1, 3), layout='constrained')

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.15])

    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    for _, spectrum in enumerate(model_outputs.keys()):
        ax.plot([model_outputs[spectrum].greedy, model_outputs[spectrum].top1, 
                 model_outputs[spectrum].top5, model_outputs[spectrum].top10], 
                marker='o', markerfacecolor='white', label=spectrum, color=colors[spectrum])
        if degeneracy_folders is not None and spectrum in degeneracy_folders: # plot expected reconstruction 
            expected = read_expected_reconstruction(degeneracy_folders[spectrum])
            ax.axhline(expected, linestyle='--', color=colors[spectrum])

    ax.grid()
    ax.set_ybound(0, 105)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(algorithms)
    ax.set_ylabel('% Sequence Reconstruction')
    #ax.set_title('Sequence Reconstruction from Spectra')
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels,loc='center', frameon=False)
    legend_ax.axis('off')
    plt.savefig(os.path.join(folders["UV-Vis + MS + NMR"], f"sequence_reconstruction_a{alpha}.svg"))  
    plt.close()

    plot_legend(handles, labels)

    return  

def plot_legend(handles, labels):
    _, ax = plt.subplots(figsize=(2, 0.5))
    ax.legend(handles=handles, labels=labels, loc='center', frameon=False, ncol=len(labels))
    ax.axis('off')
    plt.savefig(f'./reconstruction_legend.svg')
    plt.close()

    return
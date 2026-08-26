import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ------ Extract Mixtures Output ------ ##

def read_mixtures_output(folder, mean=False):

    # read in predictions 
    lengths_df = pd.read_csv(os.path.join(folder, 'L.csv'), index_col=0)
    f_D_df = pd.read_csv(os.path.join(folder, 'f_D.csv'), index_col=0)
    block_lengths_df = pd.read_csv(os.path.join(folder, 'L_block.csv'), index_col=0)
    
    if mean:
        lengths_df = lengths_df.mean()
        f_D_df = f_D_df.mean()
        block_lengths_df = block_lengths_df.mean()

    return {"L": lengths_df, "f_D": f_D_df, "L_block": block_lengths_df}

def read_mixtures_top_scores(folder, models, top=1):

    # read in predictions
    results_df = pd.read_csv(os.path.join(folder, 'all_results.csv'), index_col=0)

    # get top-kth score
    results_df = results_df.groupby('Mixture', as_index=False).nth(top-1)

    # rename columns to match model names
    results_df.drop(columns=['NMR_Predicted', 'MS_Predicted', 'UV-Vis_Predicted', 'All_Predicted'], inplace=True)
    results_df.rename(columns={'NMR_Score': models[0], 'MS_Score': models[1], 
                               'UV-Vis_Score': models[2], 'All_Score': models[3]}, inplace=True)

    return results_df

## ------ Get Mixtures Errors ------ ##

def get_mixtures_errors(mixtures_output, models):

    mixtures_errors = {}

    for key in mixtures_output:
        error_df = mixtures_output[key].copy()
        for model in models:
            error_df[model] = error_df[model] - error_df['Target']

        mixtures_errors[f'delta_{key}'] = error_df[models]

    return mixtures_errors

## ------ Plot Mixture Predictions ------ ##

def plot_mixtures_scores(folder, lambdas, models, colors, positions):

    mixtures_top_scores = {1: {}, 2: {}}

    for lamb in lambdas:
        lambda_folder = os.path.join(folder, f'lambda_{lamb}/')
        mixtures_top_scores[1][lamb] = read_mixtures_top_scores(lambda_folder, models, top=1)
        mixtures_top_scores[2][lamb] = read_mixtures_top_scores(lambda_folder, models, top=2)
    
    for top, scores in mixtures_top_scores.items():

        _, ax = plt.subplots(figsize=(6, 1.5))
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        
        for lamb in lambdas:
            vplot = ax.violinplot([scores[lamb][model] for model in models], 
                            positions=positions+lamb, widths=0.03, showmeans=True)
            
            for patch, color in zip(vplot['bodies'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(1)
            
            vplot['cmeans'].set_color('black')
            vplot['cmins'].set_color('black')
            vplot['cmaxes'].set_color('black')
            vplot['cbars'].set_color('black')

        ax.set_xlabel(r'$\lambda$')
        ax.set_xlim(lambdas[0] - 0.15, lambdas[-1] + 0.15)
        ax.set_xticks(lambdas)
        ax.set_xticklabels(lambdas)
        ax.set_ylabel(f'Top-{top} Scores')
        ax.set_ylim(-0.1, 1.1)
        #handles, labels = ax.get_legend_handles_labels()
        plt.savefig(os.path.join(folder, f'top_{top}_scores.svg'))  
        plt.close()

    #plot_legend(handles[:len(models)], labels[:len(models)])

    return  

def plot_mixtures_errors(folder, lambdas, stats, models, colors, positions):

    mixtures_errors = {}

    for lamb in lambdas:
        lambda_folder = os.path.join(folder, f'lambda_{lamb}/')
        mixtures_errors[lamb] = get_mixtures_errors(read_mixtures_output(lambda_folder), models)

    for stat, name in stats.items():

        _, ax = plt.subplots(figsize=(6, 1.2))
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.axhline(0, color='black', linestyle='-')
        
        for lamb in lambdas:
            vplot = ax.violinplot([mixtures_errors[lamb][stat][model] for model in models], 
                            positions=positions+lamb, widths=0.03, showmeans=True)
            
            for patch, color in zip(vplot['bodies'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(1)
            
            vplot['cmeans'].set_color('black')
            vplot['cmins'].set_color('black')
            vplot['cmaxes'].set_color('black')
            vplot['cbars'].set_color('black')

        ax.set_xlabel(r'$\lambda$', fontsize=6)
        ax.set_xlim(lambdas[0] - 0.15, lambdas[-1] + 0.15)
        ax.set_xticks(lambdas)
        ax.set_xticklabels(lambdas, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylabel(name, fontsize=6)
        plt.savefig(os.path.join(folder, f'{stat}.svg'))  
        plt.close()

    return  

def plot_mixtures(folder, lambdas, stats, models, colors, fullrange=False):

    mixtures_outputs = {}

    for lamb in lambdas:
        lambda_folder = os.path.join(folder, f'lambda_{lamb}/')
        mixtures_outputs[lamb] = read_mixtures_output(lambda_folder, mean=True)
        
    for stat, name in stats.items():

        fig = plt.figure(figsize=(2, 2.2))
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9]) 
        
        for model, color in zip(models, colors):
            if model == 'Target':
                ax.plot(lambdas, [mixtures_outputs[lamb][stat][model] for lamb in lambdas], 
                                    marker='o', markerfacecolor='white', linestyle=(0, (5, 1)), label=model, c=color)
            else:
                ax.plot(lambdas, [mixtures_outputs[lamb][stat][model] for lamb in lambdas], 
                        marker='o', markerfacecolor='white', label=model, c=color)

        ax.grid(axis='y')
        ax.set_xlabel(r'$\lambda$', fontsize=10)
        ax.set_xticks(lambdas)
        ax.set_xticklabels(lambdas)
        ax.set_ylabel(name, fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
    
        if fullrange:
            if name == "L":
                ax.set_ylim(3, 20)
            elif name == "f_D":
                ax.set_ylim(0, 1)
            elif name == "L_block":
                ax.set_ylim(0, 20)
            plt.savefig(os.path.join(folder, f'{stat}_fullrange.svg'))  
        else:
            if name == "L":
                ax.set_yticks([10, 11, 12, 13])
            plt.savefig(os.path.join(folder, f'{stat}.svg'))  
        plt.close()

    plot_legend(handles, labels)

    return  

def plot_legend(handles, labels, label='mixtures'):
    _, ax = plt.subplots(figsize=(2, 0.5))
    ax.legend(handles=handles, labels=labels, loc='center', frameon=False, ncol=len(labels))
    ax.axis('off')
    plt.savefig(os.path.join(f'{label}_legend.svg'))
    plt.close()

    return

## ----- Run All Mixture Plost ----- ##

# display labels
STATS = {"L": r"$\overline{{\langle L \rangle_{{mix}}}}$", 
             "f_D": r"$\overline{{\langle f_D \rangle_{{mix}}}}$", 
             "L_block": r"$\overline{{\langle L_{{block}} \rangle_{{mix}}}}$"}

ERROR_STATS = {"delta_L": r"$\langle L \rangle_{{mix}}  - \langle L \rangle_{{mix}}^{{true}}$", 
                "delta_f_D": r"$\langle f_D \rangle_{{mix}} - \langle f_D \rangle_{{mix}}^{{true}}$", 
                "delta_L_block": r"$\langle L_{{block}} \rangle_{{mix}} - \langle L_{{block}} \rangle_{{mix}}^{{true}}$"}

# violin offsets for 4 models at each lambda tick
VIOLIN_POSITIONS = np.array([-0.075, -0.025, 0.025, 0.075])

def run_mixtures_plots(folder, lambdas, colors, fullrange=False):
    models = ['NMR', 'MS', 'UV-Vis', 'UV-Vis + MS + NMR', 'Target']
    colors_list = [colors[model] if model in colors else 'black' for model in models]

    ## ---- Fig 6 ---- ##
    plot_mixtures(folder, lambdas, STATS, models, colors_list, fullrange=fullrange)
    ## ---- Fig S7 ---- ##
    plot_mixtures_errors(folder, lambdas, ERROR_STATS, models[:-1], colors_list[:-1], VIOLIN_POSITIONS)
    ## ---- Fig S8 ---- ##
    plot_mixtures_scores(folder, lambdas, models[:-1], colors_list[:-1], VIOLIN_POSITIONS)

    return

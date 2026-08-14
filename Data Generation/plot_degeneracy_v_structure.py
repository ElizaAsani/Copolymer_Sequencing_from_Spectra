import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from seq_generator import calcAvgBlockLength, calcLambda
from lempel_ziv_complexity import lempel_ziv_complexity

min_length = 3
max_length = 20

def write_sequence_stats(df):

    if 'CalcLambda' not in df.columns:
        df['CalcLambda'] = df['Sequence'].apply(lambda seq: calcLambda(seq))
    
    if 'AvgBlockLength' not in df.columns:
        df['AvgBlockLength'] = df['Sequence'].apply(lambda seq: calcAvgBlockLength(seq))

    if 'LempelZiv' not in df.columns:
        df['LempelZiv'] = df['Sequence'].apply(lambda seq: lempel_ziv_complexity(seq))

    return df

def plot_degen_v_stats_all(df):

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].scatter(df['CalcLambda'], df['Degeneracy'], alpha=0.25, s=8)
    axs[0].set_xlabel('Calculated Lambda')
    axs[0].set_xlim(-1.1, 1.1)
    axs[0].set_xticks(np.arange(-1, 1.5, 0.5))
    axs[0].set_ylabel('Degeneracy')

    max_block_length = np.ceil(df['AvgBlockLength'].max())
    axs[1].scatter(df['AvgBlockLength'], df['Degeneracy'], alpha=0.25, s=8)
    axs[1].set_xlabel('Average Block Length')
    axs[1].set_xlim(0.5, max_block_length + 0.5)
    axs[1].set_xticks(np.arange(0, max_block_length+1, 4))
    axs[1].set_ylabel('Degeneracy')

    max_lempel_ziv = np.ceil(df['LempelZiv'].max())
    axs[2].scatter(df['LempelZiv'], df['Degeneracy'], alpha=0.25, s=8)
    axs[2].set_xlabel('Lempel-Ziv Complexity')
    axs[2].set_xlim(0.5, max_lempel_ziv + 0.5)
    axs[2].set_xticks(np.arange(0, max_lempel_ziv+1, 2))
    axs[2].set_ylabel('Degeneracy')

    corr_fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[['CalcLambda', 'AvgBlockLength', 'LempelZiv', 'Degeneracy']].corr()
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar=True)

    return fig, corr_fig

def plot_degen_v_stats_dataset(df):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].scatter(df['Lambda'], df['Degeneracy'], alpha=0.25)
    axs[0, 0].set_xlabel('Lambda')
    axs[0, 0].set_xlim(-1.1, 1.1)
    axs[0, 0].set_xticks(np.arange(-1, 1.1, 0.2))
    axs[0, 0].set_ylabel('Degeneracy')

    axs[0, 1].scatter(df['CalcLambda'], df['Degeneracy'], alpha=0.25)
    axs[0, 1].set_xlabel('Calculated Lambda')
    axs[0, 1].set_xlim(-1.1, 1.1)
    axs[0, 1].set_xticks(np.arange(-1, 1.1, 0.2))
    axs[0, 1].set_ylabel('Degeneracy')

    max_block_length = np.ceil(df['AvgBlockLength'].max())
    axs[1, 0].scatter(df['AvgBlockLength'], df['Degeneracy'], alpha=0.25)
    axs[1, 0].set_xlabel('Average Block Length')
    axs[1, 0].set_xlim(0.5, max_block_length + 0.5)
    axs[1, 0].set_xticks(np.arange(1, max_block_length+1, np.ceil(max_block_length/10)))
    axs[1, 0].set_ylabel('Degeneracy')

    max_lempel_ziv = np.ceil(df['LempelZiv'].max())
    axs[1, 1].scatter(df['LempelZiv'], df['Degeneracy'], alpha=0.25)
    axs[1, 1].set_xlabel('Lempel-Ziv Complexity')
    axs[1, 1].set_xlim(0.5, max_lempel_ziv + 0.5)
    axs[1, 1].set_xticks(np.arange(1, max_lempel_ziv+1, np.ceil(max_lempel_ziv/10)))
    axs[1, 1].set_ylabel('Degeneracy')

    corr_fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[['Lambda', 'CalcLambda', 'AvgBlockLength', 'LempelZiv', 'Degeneracy']].corr()
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar=True)

    return fig, corr_fig

def plot_stats_histograms(df):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    sns.histplot(df['CalcLambda'], ax=axs[0], binwidth=0.2, stat='percent')

    max_block_length = np.ceil(df['AvgBlockLength'].max())
    sns.histplot(df['AvgBlockLength'], ax=axs[1], binwidth=1, stat='percent')
    axs[1].set_xticks(np.arange(0, max_block_length+1, 4))

    max_lempel_ziv = np.ceil(df['LempelZiv'].max())
    sns.histplot(df['LempelZiv'], ax=axs[2], binwidth=1, stat='percent')
    axs[2].set_xticks(np.arange(0, max_lempel_ziv+1, 2))

    return fig

def all():
    folder = 'Output/all/uv_vis/distinguishability/e0.01_i0.01/'
    out_folder = folder + 'degeneracy_v_stats/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    dfs = []

    for i in range(min_length, max_length + 1):
        filename = folder + f'uv_vis_degeneracies_{i}.csv'
        df = pd.read_csv(filename)
        df = write_sequence_stats(df)
        #df.to_csv(filename, index=False)
        dfs.append(df)
        """
        fig, corr_fig = plot_degen_v_stats_all(df)
        fig.suptitle(f'Sequence Length {i}')
        fig.tight_layout()
        fig.savefig(out_folder + f'degeneracy_v_stats_{i}.png')
        plt.close(fig)
        corr_fig.suptitle(f'Sequence Length {i} Correlation Matrix')
        corr_fig.savefig(out_folder + f'correlation_matrix_{i}.png')
        plt.close(corr_fig)"""

    df_all = pd.concat(dfs, ignore_index=True)
    
    fig, corr_fig = plot_degen_v_stats_all(df_all)
    fig.suptitle('All Sequence Lengths')
    fig.tight_layout()
    fig.savefig(out_folder + 'degeneracy_v_stats_all.png')
    plt.close(fig)
    corr_fig.suptitle('All Sequence Lengths Correlation Matrix')
    corr_fig.savefig(out_folder + 'correlation_matrix_all.png')
    plt.close(corr_fig)
    """
    fig = plot_stats_histograms(df_all)
    fig.tight_layout()
    fig.savefig('./Output/all/seq_stats_histograms_all.png')
    plt.close(fig)"""

    return

def dataset():
    folder = 'Output/all/nmr/reference_dataset/'
    df = pd.read_csv(folder + 'nmr_dataset_duplicates.csv')
    df = write_sequence_stats(df)
    fig, corr_fig = plot_degen_v_stats_dataset(df)
    fig.tight_layout()
    fig.savefig(folder + 'degeneracy_v_stats_dataset.png')
    plt.close(fig)
    corr_fig.savefig(folder + 'correlation_matrix_dataset.png')
    plt.close(corr_fig)

    fig = plot_stats_histograms(df)
    fig.tight_layout()
    fig.savefig('./Output/all/seq_stats_histograms_dataset.png')
    plt.close(fig)

    return

if __name__ == '__main__':
    #all()
    #dataset()
    seq1 = 'ADADADADADADADADADAD'
    seq2 = 'ADAADADADDAADADDDADD'
    print(f'Sequence 1: {seq1}, Lambda: {calcLambda(seq1)}, L_block: {calcAvgBlockLength(seq1)}, LZ Complexity: {lempel_ziv_complexity(seq1)}')
    print(f'Sequence 2: {seq2}, Lambda: {calcLambda(seq2)}, L_block: {calcAvgBlockLength(seq2)}, LZ Complexity: {lempel_ziv_complexity(seq2)}')
"""
Functions to compute different degeneracy metrics for a given sequence and its corresponding spectrum.

Spectra are considered to be degenerate if multiple sequences produce indistinguishable spectra.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Multispectra-to-sequence Transformer'))
from SequenceEncoder import stratified_split_indices

# =====================================================================
# Comparison strategies - choose one per modality
# =====================================================================

def exact_match_degeneracy(sequences, getDiscreteSpectrum, out_file=None):
    """
    Degeneracy via exact match of discrete spectra. Use for NMR/MS.
    'getDiscreteSpectrum' must return a vector of intensities.
    
    Returns a dataframe containing ['Sequence', 'Degeneracy'] for each sequence in the input list.
    """
    df = pd.DataFrame({'Sequence': sequences, 'Spectrum': [getDiscreteSpectrum(seq) for seq in sequences]})
    df['Degeneracy'] = df.groupby('Spectrum')['Sequence'].transform('count')

    result = df[['Sequence', 'Degeneracy']]

    if out_file is not None:
        result.to_csv(out_file, index=False)

    return result

def threshold_match_degeneracy(sequences, getDiscreteSpectrum, x_res, intensity_res,
                               device=None, batch_size=10000, out_file=None):
    """
    Degeneracy via pairwise threshold matching. Use for UV-Vis to account for 
    missed degeneracies due to rounding errors.

    'getDiscreteSpectrum' must return a fixed-length (x, intensity) pair and receives 
    a parameter used to pad the spectra to a maximum length.

    Returns a dataframe containing ['Sequence', 'Degeneracy'] for each sequence in the input list.
    """
    print(device)
    N = len(sequences)
    max_length = max(len(seq) for seq in sequences)

    # calculate degeneracies
    spectra = [getDiscreteSpectrum(seq, max_length) for seq in sequences]
    xs = torch.tensor(np.array([spec[0] for spec in spectra]), dtype=torch.float32, device=device)
    intensities = torch.tensor(np.array([spec[1] for spec in spectra]), dtype=torch.float32, device=device)

    degeneracy = torch.zeros((N,), dtype=torch.int, device=device)

    for i in range(0, N, batch_size):    
        x1, i1 = xs[i:i+batch_size], intensities[i:i+batch_size]
        for j in range(i, N, batch_size):
            x2, i2 = xs[j:j+batch_size], intensities[j:j+batch_size]

            # cdist computes pairwise distances between two sets of vectors; 
            #   p=inf gives the max distance b/n absorption or intensity values
            #   for a pair of spectra --> equivalent to checking if all peaks are 
            #   within the threshold
            x_sim = torch.cdist(x1, x2, p=torch.inf) < x_res
            i_sim = torch.cdist(i1, i2, p=torch.inf) < intensity_res

            s_sim = x_sim & i_sim

            degeneracy[i:i+batch_size] = degeneracy[i:i+batch_size] + torch.sum(s_sim, dim=1)

            if i != j:
                degeneracy[j:j+batch_size] = degeneracy[j:j+batch_size] + torch.sum(s_sim, dim=0)

    # generate dataframe to store sequences and their max similarities
    result = pd.DataFrame({'Sequence': sequences, 'Degeneracy': degeneracy.detach().cpu().numpy()})
    
    if out_file is not None:
        result.to_csv(out_file, index=False)

    return result

# =====================================================================
# Shared metric calculations
# =====================================================================

def all_lengths_degeneracy(getDiscreteSpectrum, degeneracy_fn, out_folder,
                           min_length, max_length, sequence_dir):
    """
    Exhaustive degeneracy analysis across all sequence lengths.

    Generates a CSV file for each sequence length containing ['Sequence', 'Degeneracy']
    for all sequences of that length.
    """

    seq_lengths, total, total_unique, largest_group = [], [], [], []

    for length in range(min_length, max_length+1):
        sequence_file = os.path.join(sequence_dir, f'all_seq_{length}.csv') 
        df = pd.read_csv(sequence_file, header=None, names=['Sequence', 'Lambda'])
        sequences = df['Sequence'].tolist()

        degeneracies_file = os.path.join(out_folder, f'degeneracies_{length}.csv')
        result = degeneracy_fn(sequences, getDiscreteSpectrum, out_file=degeneracies_file)
        
        seq_lengths.append(length)
        total.append(len(sequences))
        total_unique.append(int((result['Degeneracy'] == 1).sum()))
        largest_group.append(int(result['Degeneracy'].max()))

    summary = pd.DataFrame({'Sequence Lengths': seq_lengths, 
                            'Total # Sequences': total, 
                            'Total # Unique Spectra': total_unique, 
                            'Largest Group': largest_group})

    summary.to_csv(os.path.join(out_folder, 'degeneracy.csv'), index=False)

    return

def get_degeneracy_map(out_folder, min_length, max_length):
    """{sequence: degeneracy} map across all lengths; merges outputs
    from all_lengths_degeneracy() into a single dictionary."""

    degeneracy_map = {}

    for length in range(min_length, max_length + 1):
        degeneracy_file = os.path.join(out_folder, f'degeneracies_{length}.csv')

        if not os.path.exists(degeneracy_file):
            continue

        df = pd.read_csv(degeneracy_file)
        degeneracy_map.update(dict(zip(df['Sequence'], df['Degeneracy'])))

    return degeneracy_map

def dataset_degeneracies(out_folder, min_length, max_length, sequence_file):
    """Computes degeneracies in the train, validate, and test set using the 
    model dataset splits and the degeneracy map from all_lengths_degeneracy().
    Reports expected reconstruction accuracy based on degeneracies in test set."""

    degeneracy_map = get_degeneracy_map(out_folder, min_length, max_length)

    dataset_folder = os.path.join(out_folder, 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)

    df = pd.read_csv(sequence_file, header=None, names=['Sequence', 'Lambda'])
    df['Degeneracy'] = df['Sequence'].map(degeneracy_map).fillna(1).astype(int)

    # degeneracies for entire dataset
    df.to_csv(os.path.join(dataset_folder, 'dataset_degeneracies.csv'), index=False)

    # stats for full dataset and for train, validate, and test sets
    train_idx, val_idx, test_idx = stratified_split_indices([len(seq) for seq in df['Sequence']]) # same as model
    splits = {'dataset': df, 'train': df.iloc[train_idx], 'validate': df.iloc[val_idx], 'test': df.iloc[test_idx]}

    summary_file = os.path.join(dataset_folder, 'dataset_degeneracies.txt')
    with open(summary_file, 'w') as f:
        print(f"Dataset Sequence Lengths: {min_length} to {max_length}", file=f)

        for name, split_df in splits.items():
            num_sequences = len(split_df)
            counts = split_df['Degeneracy'].value_counts().sort_index()

            degen_distr_df = counts.reset_index()
            degen_distr_df.columns = ['Degeneracy', 'Count']
            expected_num_corr = (degen_distr_df['Count'] / degen_distr_df['Degeneracy']).sum()
            expected_percent_corr = 100 * expected_num_corr / num_sequences

            split_df.to_csv(os.path.join(dataset_folder, f'{name}_degeneracies.csv'), index=False)
            degen_distr_df.to_csv(os.path.join(dataset_folder, f'{name}_degeneracy_distribution.csv'), index=False)

            print("----------------------------------", file=f)
            print(name.capitalize() + " set" if (name != 'dataset') else name.capitalize(), file=f)
            print(f"Total # Sequences: {num_sequences}", file=f)
            print(f"Total # Unique Spectra: {(split_df['Degeneracy'] == 1).sum()}", file=f)
            print(f"Largest Group: {split_df['Degeneracy'].max()}", file=f)
            print(f"Expected # Correctly Reconstructed: {expected_num_corr:.2f}", file=f)
            print(f"Expected % Correctly Reconstructed: {expected_percent_corr:.2f}%", file=f)
        print("----------------------------------", file=f)

    return

def degeneracy_aware_reconstruction(isIndistinguishable, model_out_folder):
    """
    Computes degeneracy-aware reconstruction accuracy for a given model's predictions.

    'isIndistinguishable' must take in two sequences and return True if their spectra are 
    indistinguishable and False otherwise. 
    """

    error_file = os.path.join(model_out_folder, 'errors.csv')
    error_df = pd.read_csv(error_file)

    error_df['Degenerate'] = [isIndistinguishable(target, predicted) for target, predicted 
                              in zip(error_df['target sequence'], error_df['predicted sequence'])]
    error_df.to_csv(os.path.join(model_out_folder, 'degenerate_errors.csv'), index=False)

    # write mean
    with open(os.path.join(model_out_folder, 'degeneracy_aware_accuracy.txt'), 'w') as f:
        print(f"Degeneracy-aware reconstruction accuracy: {error_df['Degenerate'].mean()}", file=f)

    return

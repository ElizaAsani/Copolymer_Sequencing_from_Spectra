"""
Mass spectra degeneracy analysis
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from degeneracy_functions import dataset_degeneracies

# =====================================================================
# Shared metric calculations
# =====================================================================

def all_lengths_degeneracy(spectra, out_folder, min_length, max_length):
    """
    Combined exhaustive degeneracy analysis across all sequence lengths.

    Generates a CSV file for each sequence length containing ['Sequence', 'Degeneracy']
    for all sequences of that length.
    """

    seq_lengths, total, total_unique, largest_group = [], [], [], []

    for length in range(min_length, max_length+1):

        spectra_degen_dfs = {}
        for spec in spectra:
            degeneracy_file = os.path.join(spec, f'degeneracies_{length}.csv')
            if not os.path.exists(degeneracy_file):
                raise FileNotFoundError(f"Degeneracy file for {spec} at length {length} not found: {degeneracy_file}")
            spectra_degen_dfs[spec] = pd.read_csv(degeneracy_file, header=0, names=['Sequence', 'Degeneracy'])
            # change column name to avoid conflicts during merge
            spectra_degen_dfs[spec].rename(columns={'Degeneracy': f'Degeneracy_{spec}'}, inplace=True)

        # merge dataframes on 'Sequence'
        merged_df = spectra_degen_dfs[spectra[0]]
        for i in range(1, len(spectra)):
            merged_df = pd.merge(merged_df, spectra_degen_dfs[spectra[i]], on='Sequence', how='inner')

        # degeneracy of combined spectra is, at most, the minimum degeneracy across all spectra
        merged_df['Degeneracy'] = merged_df[[f'Degeneracy_{spec}' for spec in spectra]].min(axis=1)

        degeneracies_file = os.path.join(out_folder, f'degeneracies_{length}.csv')
        merged_df.to_csv(degeneracies_file, index=False)      

        seq_lengths.append(length)
        total.append(len(merged_df))  
        total_unique.append(int((merged_df['Degeneracy'] == 1).sum()))  
        largest_group.append(merged_df['Degeneracy'].max())

    summary = pd.DataFrame({'Sequence Lengths': seq_lengths, 
                            'Total # Sequences': total, 
                            'Total # Unique Spectra': total_unique, 
                            'Largest Group': largest_group})

    summary.to_csv(os.path.join(out_folder, 'degeneracy.csv'), index=False)

    return

MIN_LENGTH, MAX_LENGTH = 3, 20
SEQUENCE_FILE = os.path.join('..', f'seq_{MIN_LENGTH}-{MAX_LENGTH}.csv')

SPECTRA = ['uv_vis', 'ms', 'nmr']
OUT_FOLDER = SPECTRA[0]
for i in range(1, len(SPECTRA)):
    OUT_FOLDER += '+' + SPECTRA[i]

def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    all_lengths_degeneracy(SPECTRA, OUT_FOLDER, MIN_LENGTH, MAX_LENGTH)

    dataset_degeneracies(OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_FILE)

if __name__ == "__main__":
    main()
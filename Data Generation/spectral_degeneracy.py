"""
Copolymer Input Generation
"""

import ast
from collections import defaultdict
import os
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from UV_Vis import *
from NMR import *
from MS import *

#------------UV-Vis Parameters-----------#
R = 5 * 10**-10             # distance between monomers [m] (5 A)
eps_r = 1.0                 # relative permittivity (1.0)
J_DD = UV_Vis.calculateJDD(R, eps_r)  # dipole-dipole coupling coefficient [eV] (-0.01)

monomers = ['D', 'A']  # monomer types (['D', 'A'])

frenkel_params = FrenkelParameters(
    monomers = monomers,      # monomer types (['D', 'A'])    
    eps = [5.0, 4.5],            # excited state energy for each monomer [eV] ([5.0, 4.5])
    mu = [10.0, 10.0],           # transition dipole moment for each monomer ([10.0, 10.0])
    J_DD = J_DD,                # dipole-dipole coupling coefficient [eV] (-0.01)
    J_SE = -0.7                  # nearest-neighbor superexchange coupling coefficient [eV] (-0.7)
)

uv_vis_plot_params = GaussianPlotParameters(
    points = 220,          # number of points on the curve to calculate (20)
    std_dev = 0.1,           # standard deviation (0.2)
    energyRange = [0.4, 6.5],    # energy range to calculate absorption over ([0, 5]))
)

#------------NMR Parameters-----------#
trimer_file = 'HNMR_trimers.csv'  # file with trimer data
dimer_file = 'HNMR_dimers.csv'    # file with dimer data

hnmr_plot_params = NMRPlotParameters(
    points = 500,          # number of points on the curve to calculate (1000)
    half_width = 0.1,      # half-width of the Lorentzian peak (0.02)
    shift_range = [21.7, 31.7], # chemical shift range in ppm ([22, 32])
    reference_shift = 31.7, # reference chemical shift in ppm (31.7)
    tolerance = 0            # tolerance for peak consolidation (0)
)

#------------Mass Spec Parameters-----------#
ms_parameters = MassSpecParameters(
        monomers = monomers + ['*'],
        formulas = [{'C':9, 'O':2, 'S':1, 'H':10}, {'C':7, 'N':3, 'H':5}, {'C':1, 'H':3}]
    )

ms_plot_parameters = BarPlotParameters(
        massRange = [140, 3700],
        binWidth = 10
    )

ms_noise_parameters = MSNoiseParameters(
        extra_peaks = 0,
        width = 1,
        weight = 1,
    )

min_length = 3
max_length = 20
folder = f"Output/all/"

def ms():

    # initialize spectrum generators
    ms_spectrum_generator = MS(ms_parameters, ms_plot_parameters, ms_noise_parameters)

    # create folder
    spec_type = 'ms'
    out_folder = folder + f'{spec_type}/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # function for spectral generation
    funct = lambda seq: getDiscreteMS(seq, ms_spectrum_generator)

    # run degeneracy calculation for all sequence lengths
    #all_lengths_degeneracy(spec_type, funct, out_folder)

    # run degeneracy calculation for dataset
    #isolated_dataset_degeneracies(spec_type, funct, out_folder)

    # get dataset degeneracies from reference
    #reference_dataset_degeneracies(spec_type, out_folder)

    # calculate degeneracy-aware reconstruction
    degeneracy_aware_reconstruction(spec_type, funct, out_folder, 'ep0_w1_d0')

def nmr():

    # initialize spectrum generators
    nmr_spectrum_generator = NMR(trimer_file, dimer_file, hnmr_plot_params)

    # create folder
    spec_type = 'nmr'
    out_folder = folder + f'{spec_type}/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # function for spectral generation
    funct = lambda seq: getDiscreteNMR(seq, nmr_spectrum_generator)

    # run degeneracy calculation for all sequence lengths
    #all_lengths_degeneracy(spec_type, funct, out_folder)

    # run degeneracy calculation for dataset
    #isolated_dataset_degeneracies(spec_type, funct, out_folder)

    # get dataset degeneracies from reference
    #reference_dataset_degeneracies(spec_type, out_folder)

    # calculate degeneracy-aware reconstruction
    degeneracy_aware_reconstruction(spec_type, funct, out_folder, 'hwhm0.1')

def all_lengths_degeneracy(spec_type, getDiscreteSpectrum, out_folder):
    """Computes, saves, and plots the degeneracy of discrete spectra for all sequence lengths.

    Inputs: 
        spec_type: string label of spectral type
        getDiscreteSpectrum: function that generates a discrete spectrum from a sequence
    """
    
    seq_lengths = []
    total = []
    total_unique = []
    largest_group = []

    for length in range(min_length, max_length+1):
        # read in sequences
        sequence_file = folder + f'all_sequences_{length}.csv' 
        df = pd.read_csv(sequence_file).drop(columns=['Lambda'])

        # get duplicates
        duplicates_file = out_folder + f'{spec_type}_duplicates_{length}.csv' 
        num_sequences, num_unique, largest = discrete_spectrum_degeneracy(df['Sequence'].tolist(), duplicates_file, getDiscreteSpectrum)
        
        # append results
        seq_lengths.append(length)
        total.append(num_sequences)
        total_unique.append(num_unique)
        largest_group.append(largest)

        # save degeneracies
        degeneracy_map = get_degeneracy_map(spec_type, out_folder, lengths=[length])
        dataset_degeneracies_file = out_folder + f'{spec_type}_degeneracie_{length}.csv'

        # add degeneracy column
        df['Degeneracy'] = df['Sequence'].map(degeneracy_map).fillna(1).astype(int)

        # save dataset duplicates
        df.to_csv(dataset_degeneracies_file, index=False)
    
    out_file = out_folder + f'{spec_type}_degeneracies.csv'
    summary = pd.DataFrame({'Sequence Lengths': seq_lengths, 
                            'Total # Sequences': total, 
                            'Total # Unique Spectra': total_unique, 
                            'Largest Group': largest_group})

    summary.to_csv(out_file)

    percents = [round(num_unique / num_total, 2)*100 for num_unique, num_total in zip(total_unique, total)]
    _, axs = plt.subplots(1, 3, figsize=(16, 6))

    axs[0].plot(seq_lengths, total, label='Total')
    axs[0].plot(seq_lengths, total_unique, label='Unique')
    axs[0].set_xticks(np.arange(4, max_length+1, 2))
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('# Sequences')
    axs[0].set_title('Number of Unique Spectra')
    axs[0].legend()
    
    axs[1].plot(seq_lengths, percents, marker='o')
    axs[1].set_xticks(np.arange(4, max_length+1, 2))
    axs[1].set_yticks(np.arange(0, 100+1, 20))
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('% Unique Spectra')
    axs[1].set_title('Percent of Unique Spectra')

    axs[2].plot(seq_lengths, largest_group, marker='o')
    axs[2].set_xticks(np.arange(4, max_length+1, 2))
    axs[2].set_xlabel('Sequence Length')
    axs[2].set_ylabel('Number of Sequences')
    axs[2].set_title('Largest Group of Degenerate Spectra')

    plt.savefig(out_folder + f'{spec_type}_unique.png')

    return

def isolated_dataset_degeneracies(spec_type, getDiscreteSpectrum, out_folder):
    """Computes, saves, and plots the degeneracy of discrete spectra for a datasest.

    Inputs: 
        spec_type: string label of spectral type
        getDiscreteSpectrum: function that generates a discrete spectrum from a sequence
    """
    out_folder = out_folder + f'/isolated_dataset/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_file = out_folder + f'{spec_type}_dataset_degeneracies.txt'
    with open(out_file, 'w') as f:
        sequence_file = f'seq_{min_length}-{max_length}.csv'

        with open(sequence_file, 'r') as sequence_file:
                reader = csv.reader(sequence_file)
                sequences, _ = zip(*[(row[0], float(row[1])) for row in reader])

        duplicates_file = out_folder + f'{spec_type}_dataset_duplicates.csv'
        num_sequences, num_unique, largest = discrete_spectrum_degeneracy(sequences, duplicates_file, getDiscreteSpectrum)

        print("----------------------------------", file=f)
        print(f"Dataset Sequence Lengths: {min_length}-{max_length}", file=f)
        print("Full dataset", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print("----------------------------------", file=f)

        # get dataset splits used in model
        """
        N = len(sequences)
        train_size = int(0.8 * N)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(N, generator=generator)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_sequences = [sequences[i] for i in train_indices]
        test_sequences  = [sequences[i] for i in test_indices]"""

        train_idx, _, test_idx = get_dataset_splits([len(seq) for seq in sequences])
        train_sequences = [sequences[i] for i in train_idx]
        test_sequences  = [sequences[i] for i in test_idx]

        train_duplicates_file = out_folder + f'{spec_type}_train_duplicates.csv'
        num_sequences, num_unique, largest = discrete_spectrum_degeneracy(train_sequences, train_duplicates_file, getDiscreteSpectrum)

        print("Training set", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print("----------------------------------", file=f)

        test_duplicates_file = out_folder + f'{spec_type}_test_duplicates.csv'
        num_sequences, num_unique, largest = discrete_spectrum_degeneracy(test_sequences, test_duplicates_file, getDiscreteSpectrum)
        
        print("Test set", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print("----------------------------------", file=f)
    
    return

def reference_dataset_degeneracies(spec_type, out_folder):

    degeneracy_map = get_degeneracy_map(spec_type, out_folder)

    # get dataset degeneracies from reference
    out_folder = out_folder + f'/reference_dataset/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    out_file = out_folder + f'{spec_type}_dataset_degeneracies.txt'
    with open(out_file, 'w') as f:
        sequence_file = f'seq_{min_length}-{max_length}.csv'
        dataset_duplicates_file = out_folder + f'{spec_type}_dataset_duplicates.csv'

        df = pd.read_csv(sequence_file, header=None, names=['Sequence', 'Lambda'])

        # add degeneracy column
        df['Degeneracy'] = df['Sequence'].map(degeneracy_map).fillna(1).astype(int)

        # save dataset duplicates
        df.to_csv(dataset_duplicates_file, index=False)

        # calculate distribution of degeneracies
        counts = df['Degeneracy'].value_counts().sort_index()
        dist_df = counts.reset_index()
        dist_df.columns = ['Degeneracy', 'Count']
        dist_df.to_csv(out_folder + f'{spec_type}_dataset_degeneracy_distribution.csv', index=False)

        # calculate metrics
        num_sequences = len(df)
        num_unique = (df['Degeneracy'] == 1).sum()
        largest = df['Degeneracy'].max()
        weighted_sum = (dist_df['Count'] / dist_df['Degeneracy']).sum()

        print("----------------------------------", file=f)
        print(f"Dataset Sequence Lengths: {min_length}-{max_length}", file=f)
        print("Full dataset", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print(f"Weighted Sum: {weighted_sum}", file=f)        
        print("----------------------------------", file=f)

        # get dataset splits used in model
        """
        N = num_sequences
        train_size = int(0.8 * N)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(N, generator=generator)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]"""

        train_indices, val_indices, test_indices = get_dataset_splits([len(seq) for seq in df['Sequence']])

        train_df = df.iloc[train_indices]
        val_df   = df.iloc[val_indices]
        test_df  = df.iloc[test_indices]

        train_duplicates_file = out_folder + f'{spec_type}_train_duplicates.csv'
        train_df.to_csv(train_duplicates_file, index=False)
        num_sequences = len(train_df)
        num_unique = (train_df['Degeneracy'] == 1).sum()
        largest = train_df['Degeneracy'].max()

        train_counts = train_df['Degeneracy'].value_counts().sort_index()
        train_dist_df = train_counts.reset_index()
        train_dist_df.columns = ['Degeneracy', 'Count']
        train_dist_df.to_csv(out_folder + f'{spec_type}_train_degeneracy_distribution.csv', index=False)
        train_weighted_sum = (train_dist_df['Count'] / train_dist_df['Degeneracy']).sum()

        print("Training set", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print(f"Weighted Sum: {train_weighted_sum}", file=f)
        print("----------------------------------", file=f)

        val_duplicates_file = out_folder + f'{spec_type}_val_duplicates.csv'
        val_df.to_csv(val_duplicates_file, index=False)
        num_sequences = len(val_df)
        num_unique = (val_df['Degeneracy'] == 1).sum()
        largest = val_df['Degeneracy'].max()

        val_counts = val_df['Degeneracy'].value_counts().sort_index()
        val_dist_df = val_counts.reset_index()
        val_dist_df.columns = ['Degeneracy', 'Count']
        val_dist_df.to_csv(out_folder + f'{spec_type}_val_degeneracy_distribution.csv', index=False)
        val_weighted_sum = (val_dist_df['Count'] / val_dist_df['Degeneracy']).sum()

        print("Validation set", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print(f"Weighted Sum: {val_weighted_sum}", file=f)
        print("----------------------------------", file=f)

        test_duplicates_file = out_folder + f'{spec_type}_test_duplicates.csv'
        test_df.to_csv(test_duplicates_file, index=False)
        num_sequences = len(test_df)
        num_unique = (test_df['Degeneracy'] == 1).sum()
        largest = test_df['Degeneracy'].max()

        test_counts = test_df['Degeneracy'].value_counts().sort_index()
        test_dist_df = test_counts.reset_index()
        test_dist_df.columns = ['Degeneracy', 'Count']
        test_dist_df.to_csv(out_folder + f'{spec_type}_test_degeneracy_distribution.csv', index=False)
        test_weighted_sum = (test_dist_df['Count'] / test_dist_df['Degeneracy']).sum()

        print("Test set", file=f)
        print(f"Total # Sequences: {num_sequences}", file=f)
        print(f"Total # Unique Spectra: {num_unique}", file=f)
        print(f"Largest Group: {largest}", file=f)
        print(f"Weighted Sum: {test_weighted_sum}", file=f)
        print("----------------------------------", file=f)

        # plot distribution of degeneracies
        plt.figure()

        all_idx = counts.index

        train_counts = train_counts.reindex(all_idx, fill_value=0)
        val_counts   = val_counts.reindex(all_idx, fill_value=0)
        test_counts  = test_counts.reindex(all_idx, fill_value=0)

        plt.bar(all_idx, train_counts, label="Train")
        plt.bar(all_idx, val_counts, bottom=train_counts, label="Validation")
        plt.bar(all_idx, test_counts, bottom=train_counts + val_counts, label="Test")

        plt.xlabel("Degeneracy")
        plt.ylabel("Count")
        plt.title("Degeneracy Distribution")
        plt.legend()

        plt.savefig(out_folder + "degeneracy_stacked.png", dpi=300)

    return

def degeneracy_aware_reconstruction(spec_type, getDiscreteSpectrum,out_folder, model_folder):
    
    # get model output
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sibling_path = os.path.join(current_dir, '..', f'Multispectra-to-sequence Transformer/Output/{spec_type}/{model_folder}/errors')
    sibling_path = os.path.normpath(sibling_path)

    error_file = sibling_path + '/errors.csv'
    error_df = pd.read_csv(error_file)

    # create out folder
    out_folder = out_folder + f'{model_folder}/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    error_df['degenerate'] = [np.array_equal(getDiscreteSpectrum(target), getDiscreteSpectrum(pred))
                              for target, pred in zip(error_df['target sequence'], error_df['predicted sequence'])]

    error_df.to_csv(out_folder + f'errors.csv', index=False)

    # write mean
    with open(sibling_path + f'/degeneracy_recon_acc.txt', 'w') as f:
        print(f"Degeneracy-aware reconstruction accuracy: {error_df['degenerate'].mean()}", file=f)
    

    return

def get_degeneracy_map(spec_type, folder, lengths=range(min_length, max_length+1)):
    
    # get reference degeneracy map
    degeneracy_map = {}

    for i in lengths:
        duplicates_i_path = folder + f'{spec_type}_duplicates_{i}.csv'
        # check if file exists
        if not os.path.exists(duplicates_i_path):
            continue

        duplicates_i = pd.read_csv(duplicates_i_path)

        for _, row in duplicates_i.iterrows():
            seq_list = ast.literal_eval(row['Sequences'])
            multiplicity = row['Length']
            for seq in seq_list:
                degeneracy_map[seq] = multiplicity

    return degeneracy_map

def get_dataset_splits(sequence_lengths):
    
    train_frac, val_frac = 0.7, 0.15

    g = torch.Generator().manual_seed(42)

    buckets = defaultdict(list)

    for idx, L in enumerate(sequence_lengths):
        buckets[int(L)].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for L, idxs in buckets.items():
        idxs = torch.tensor(idxs, dtype=torch.long)

        # shuffle using torch permutation
        perm = torch.randperm(len(idxs), generator=g)
        idxs = idxs[perm]

        n = len(idxs)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    # final shuffle per split
    def shuffle_list(x):
        x = torch.tensor(x, dtype=torch.long)
        return x[torch.randperm(len(x), generator=g)]

    train_idx = shuffle_list(train_idx)
    val_idx = shuffle_list(val_idx)
    test_idx = shuffle_list(test_idx)

    return train_idx, val_idx, test_idx

    
def discrete_spectrum_degeneracy(sequences, duplicates_file, getDiscreteSpectrum):
    """Computes the degeneracy of discrete spectra by a 1:1 matching at each point; if degenerate spectra exist, 
        the groups of sequences are written to a file.

    Inputs: 
        sequences: list containing sequences to compare
        getDiscreteSpectrum: function that generates a discrete spectrum from a sequence

    Returns:
        num_sequences: the total number of sequences in the file
        num_unique: the total number of sequences with unique spectra
        largest: the size of the largest group of sequences with degenerate spectra
    """
    
    num_sequences = len(sequences) 

    spectra = {}

    for seq in sequences:
        spectra[seq] = getDiscreteSpectrum(seq)
        
    spectra = pd.DataFrame(list(spectra.items()), columns=['Sequence', 'Spectrum'])
    duplicates, duplicate_groups = getDuplicateSpectra(spectra)

    num_unique = num_sequences - len(duplicates)
    if len(duplicates) == 0:
        largest = 1
    else:
        largest = duplicate_groups['Length'][0]
        duplicate_groups.to_csv(duplicates_file)

    return num_sequences, num_unique, largest

def getDiscreteMS(sequence, ms_spectrum_generator):
    # returns a tuple of the mass fragments for a given sequence
    return tuple(ms_spectrum_generator.generateMassSpectrum(sequence).keys())

def getDiscreteNMR(sequence, nmr_spectrum_generator):
    # returns a tuple of the trimer and endgroup counts for a given sequence
    trimers = nmr_spectrum_generator._countTrimers(sequence)
    endgroups = nmr_spectrum_generator._countEndgroups(sequence)
    subgroups = trimers | endgroups
    return tuple(subgroups.values())

def getDiscreteEigenvalues(sequence, uv_vis_spectrum_generator, rounding=0.01):
    # returns a tuple of rounded absorption energies for a given sequence
    return tuple(round(uv_vis_spectrum_generator.getAbsorption(sequence)['energy'] / rounding) * rounding)

def getDiscreteUVVIS(sequence, uv_vis_spectrum_generator, rounding=(0.01, 0.01)):
    # returns a tuple of rounded absorption energies and intensities for a given sequence
    absorption = uv_vis_spectrum_generator.getAbsorption(sequence)
    energy, intensity = np.array(absorption['energy']), np.array(absorption['intensity'])
    intensity = intensity / intensity.max()  # normalize intensity to max value

    energy_res, intensity_res = rounding
    
    if energy_res > 0:
        energy = np.round(energy / energy_res) * energy_res
    if intensity_res > 0:
        intensity = np.round(intensity / intensity_res) * intensity_res

    # drop zero-intensity values
    non_zero = intensity != 0
    energy = energy[non_zero]
    intensity = intensity[non_zero]
    
    return tuple(np.concat([energy, intensity]))

def getDuplicateSpectra(spectra_df):
    """Returns the sequences with identical spectra."""

    duplicates = spectra_df[spectra_df.duplicated(subset=['Spectrum'], keep=False)] 

    duplicate_groups = (
        duplicates
        .groupby('Spectrum')['Sequence']
        .apply(list)                 # collect sequences in each group
        .reset_index(drop=True)      # drop the spectrum key
        .to_frame(name='Sequences')  # make dataframe
    )

    duplicate_groups['Length'] = duplicate_groups['Sequences'].apply(len)

    duplicate_groups = duplicate_groups.sort_values('Length', ascending=False).reset_index(drop=True)

    return duplicates, duplicate_groups



if __name__ == '__main__':
    #nmr()
    ms()
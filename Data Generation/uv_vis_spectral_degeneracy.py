"""
Copolymer Input Generation
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from spectral_degeneracy import get_dataset_splits
from UV_Vis import *

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
    energyRange = [0.4, 6.5]    # energy range to calculate absorption over ([0, 5]))
)

unit = 'energy' if uv_vis_plot_params.energyRange is not None else 'wavelength'
x_res = 0.01 if unit == 'energy' else 1
intensity_res = 0.01

min_length = 3
max_length = 20
folder = f"Output/all/"

# initialize spectrum generators
uv_vis_spectrum_generator = UV_Vis(frenkel_params, uv_vis_plot_params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():

    # create folder
    spec_type = 'uv_vis'
    if unit == 'energy':
        out_folder = folder + f'{spec_type}/distinguishability/e{x_res}_i{intensity_res}/'
    else:
        out_folder = folder + f'{spec_type}/distinguishability/w{x_res}_i{intensity_res}/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # run degeneracy calculation for all sequence lengths
    #all_lengths_degeneracy(spec_type, out_folder)

    # get dataset degeneracies from reference
    #reference_dataset_degeneracies(spec_type, out_folder)

    # calculate degeneracy-aware reconstruction
    degeneracy_aware_reconstruction(spec_type, getDiscreteUVVIS, out_folder, f'std0.1')

def all_lengths_degeneracy(spec_type, out_folder):
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
        sequence_file = folder + f'all_sequences_{length}.csv' 

        with open(sequence_file, 'r') as sequence_file:
            reader = csv.reader(sequence_file)
            sequences, _ = zip(*[(row[0], float(row[1])) for row in reader])

        degeneracies_file = out_folder + f'{spec_type}_degeneracies_{length}.csv'

        num_sequences, num_unique, largest = discrete_spectrum_degeneracy(sequences, degeneracies_file)
        
        seq_lengths.append(length)
        total.append(num_sequences)
        total_unique.append(num_unique)
        largest_group.append(largest)

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

def reference_dataset_degeneracies(spec_type, out_folder):

    # get reference degeneracy map
    degeneracy_map = {}

    for i in range(min_length, max_length + 1):
        degeneracies_i_path = out_folder + f'{spec_type}_degeneracies_{i}.csv'
        # check if file exists
        if not os.path.exists(degeneracies_i_path):
            continue

        degeneracies_i = pd.read_csv(degeneracies_i_path)

        for _, row in degeneracies_i.iterrows():
            degeneracy_map[row['Sequence']] = row['Degeneracy']

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

def degeneracy_aware_reconstruction(spec_type, getDiscreteSpectrum, out_folder, model_folder):
    
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

    error_df['degenerate'] = [isIndistinguishable(getDiscreteSpectrum(target), getDiscreteSpectrum(pred))
                              for target, pred in zip(error_df['target sequence'], error_df['predicted sequence'])]

    error_df.to_csv(out_folder + f'errors.csv', index=False)

    # write mean
    with open(sibling_path + f'/degeneracy_recon_acc.txt', 'w') as f:
        print(f"Degeneracy-aware reconstruction accuracy: {error_df['degenerate'].mean()}", file=f)
    
    return

def discrete_spectrum_degeneracy(sequences, out_file, batch_size = 2000):

    N = len(sequences)
    max_length = max(len(seq) for seq in sequences)

    # calculate degeneracies
    spectra = [getDiscreteUVVIS(seq, max_length) for seq in sequences]
    xs = torch.tensor(np.array([spec[0] for spec in spectra]), dtype=torch.float32, device=device)
    intensities = torch.tensor(np.array([spec[1] for spec in spectra]), dtype=torch.float32, device=device)

    degeneracy = torch.zeros((N,), dtype=torch.int, device=device)

    for i in range(0, N, batch_size):    
        x1, i1 = xs[i:i+batch_size], intensities[i:i+batch_size]
        for j in range(i, N, batch_size):
            x2, i2 = xs[j:j+batch_size], intensities[j:j+batch_size]

            x_sim = torch.cdist(x1, x2, p=torch.inf) < x_res
            i_sim = torch.cdist(i1, i2, p=torch.inf) < intensity_res

            s_sim = x_sim & i_sim

            degeneracy[i:i+batch_size] = degeneracy[i:i+batch_size] + torch.sum(s_sim, dim=1)

            if i != j:
                degeneracy[j:j+batch_size] = degeneracy[j:j+batch_size] + torch.sum(s_sim, dim=0)

    # generate dataframe to store sequences and their max similarities
    indistinguishability_df = pd.DataFrame({'Sequence': sequences,
                                  'Degeneracy': degeneracy.detach().cpu().numpy()})
    
    num_unique = torch.sum(degeneracy == 1).item()
    largest = torch.max(degeneracy).item()

    indistinguishability_df.to_csv(out_file, index=False)

    return N, num_unique, largest
    
def getDiscreteUVVIS(sequence, max_length=max_length):
    # returns a tuple of rounded absorption energies and intensities for a given sequence
    absorption = uv_vis_spectrum_generator.getAbsorption(sequence)
    if unit == 'energy':
        x = np.array(absorption['energy'])
    elif unit == 'wavelength': 
        x = np.array(absorption['wavelength'])
    else:
        raise ValueError("Invalid unit. Must be 'energy' or 'wavelength'.")

    intensity = np.array(absorption['intensity'])
    intensity = intensity / intensity.max()  # normalize intensity to max value

    # drop zero-intensity values
    non_zero = intensity >= intensity_res
    x = x[non_zero]
    intensity = intensity[non_zero]

    # pad to max length
    x = np.pad(x, (0, max_length - len(x)), constant_values=0)
    intensity = np.pad(intensity, (0, max_length - len(intensity)), constant_values=0)

    return x, intensity

def isIndistinguishable(spec1, spec2):
    """Returns True if two spectra are indistinguishable, False otherwise."""

    x1, intensity1 = spec1
    x2, intensity2 = spec2

    return (np.all(np.abs(x1 - x2) < x_res) and
            np.all(np.abs(intensity1 - intensity2) < intensity_res))

if __name__ == '__main__':
    main()
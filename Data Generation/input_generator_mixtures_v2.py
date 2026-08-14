"""
Copolymer Input Generation
"""

import csv
import ast
import os
import h5py
import numpy as np
from UV_Vis import *
from NMR import *
from MS import *
import matplotlib.pyplot as plt

#------------UV-Vis Parameters-----------#
R = 5 * 10**-10                         # distance between monomers [m] (5 A)
eps_r = 1.0                             # relative permittivity (1.0)
J_DD = UV_Vis.calculateJDD(R, eps_r)    # dipole-dipole coupling coefficient [eV] (-0.01)

monomers = ['D', 'A']  # monomer types (['D', 'A'])

frenkel_params = FrenkelParameters(
    monomers = monomers,        # monomer types (['D', 'A'])    
    eps = [5.0, 4.5],           # excited state energy for each monomer [eV] ([5.0, 4.5])
    mu = [10.0, 10.0],          # transition dipole moment for each monomer ([10.0, 10.0])
    J_DD = J_DD,                # dipole-dipole coupling coefficient [eV] (-0.01)
    J_SE = -0.7                 # nearest-neighbor superexchange coupling coefficient [eV] (-0.7)
)

uv_vis_plot_params = GaussianPlotParameters(
    points = 220,                   # number of points on the curve to calculate (220)
    std_dev = 0.1,                  # standard deviation (0.1)
    energyRange = [0.4, 6.5]       # energy range to calculate absorption over ([0.4, 6.5]))
)

#------------NMR Parameters-----------#
trimer_file = 'HNMR_trimers.csv'  # file with trimer data
dimer_file = 'HNMR_dimers.csv'    # file with dimer data

nmr_plot_params = NMRPlotParameters(
    points = 500,                   # number of points on the curve to calculate (500)
    half_width = 0.1,               # half-width of the Lorentzian peak (0.1)
    shift_range = [21.7, 31.7],     # chemical shift range in ppm ([22, 32])
    reference_shift = 31.7,         # reference chemical shift in ppm (31.7)
    tolerance = 0                   # tolerance for peak consolidation (0)
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
        dropout = 0,
        extra_peaks = 0,
        width = 1,
        weight = 1,
    )

lambdas = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]

# initialize spectrum generators
uv_vis_spectrum_generator = UV_Vis(frenkel_params, uv_vis_plot_params)
nmr_spectrum_generator = NMR(trimer_file, dimer_file, nmr_plot_params)
ms_spectrum_generator = MS(ms_parameters, ms_plot_parameters, ms_noise_parameters)

def main():

    for lamb in lambdas:
        filename = f'./Output/datasets/mixtures/mixtures_L10_lamb{lamb}_NOISE0.h5'

        mixtures_file = f'mixtures_L10_lamb{lamb}.csv'  # file with sequences and lambdas
        reader = csv.reader(open(mixtures_file, 'r', encoding="utf-8-sig"))
        mixtures = []
        ratios = []
        lambdas = []
        for row in reader:
            sequences = ast.literal_eval(row[0])
            ratio = ast.literal_eval(row[1])
            lambd = float(row[2])
            mixtures.append(sequences)
            ratios.append(ratio)
            lambdas.append(lambd)

        num_mixtures = len(mixtures)      
        
        # open and write to file
        with h5py.File(filename, 'w') as f:
            
            uv_vis_ds = f.create_dataset(
                'uv_vis',
                shape=(num_mixtures, uv_vis_plot_params.points),
                dtype=np.float32,
                chunks=(1, uv_vis_plot_params.points)
            )
            uv_vis_ds.attrs['points'] = uv_vis_plot_params.points
            uv_vis_ds.attrs['energy_range'] = uv_vis_plot_params.energyRange
            uv_vis_ds.attrs['std_dev'] = uv_vis_plot_params.std_dev
            if uv_vis_spectrum_generator.peaks is not None:
                uv_vis_ds.attrs['peaks'] = uv_vis_spectrum_generator.peaks

            nmr_ds = f.create_dataset(
                'nmr',
                shape=(num_mixtures, nmr_plot_params.points),
                dtype=np.float32,       
                chunks=(1, nmr_plot_params.points)
            )
            nmr_ds.attrs['points'] = nmr_plot_params.points
            nmr_ds.attrs['shift_range'] = nmr_spectrum_generator.shift_range #nmr_plot_params.shift_range
            nmr_ds.attrs['half_width'] = nmr_plot_params.half_width

            ms_ds = f.create_dataset(
                'ms',
                shape=(num_mixtures, len(ms_spectrum_generator.x)),
                dtype=np.float32,
                chunks=(1, len(ms_spectrum_generator.x))
            )
            ms_ds.attrs['points'] = len(ms_spectrum_generator.x)
            ms_ds.attrs['mass_range'] = ms_plot_parameters.massRange
            ms_ds.attrs['bin_width'] = ms_plot_parameters.binWidth
            ms_ds.attrs['dropout'] = ms_noise_parameters.dropout
            ms_ds.attrs['extra_peaks'] = ms_noise_parameters.extra_peaks
            ms_ds.attrs['noise_width'] = ms_noise_parameters.width
            ms_ds.attrs['peak_weight'] = ms_noise_parameters.weight

            sequence_ds = f.create_dataset(
                'sequence',
                shape=(num_mixtures,),
                dtype=h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8')),
                chunks=(1,)
            )
            sequence_ds.attrs['monomers'] = monomers
            sequence_ds.attrs['max_length'] = 20

            ratio_ds = f.create_dataset(
                'ratio',
                shape=(num_mixtures,),
                dtype=h5py.vlen_dtype(np.float32),
                chunks=(1,)
            )

            lambda_ds = f.create_dataset(
                'lambda',
                shape=(num_mixtures,),
                dtype=np.float32,
                chunks=(1,)
            )
            
            for i in range(num_mixtures):
                # get sequence and lambda
                sequences = mixtures[i]
                ratio = ratios[i]
                lambd = lambdas[i]

                # store the combined spectra
                uv_vis_ds[i, :] = get_mixed_uv_vis_spectrum(sequences, ratio, uv_vis_spectrum_generator)
                nmr_ds[i, :] = get_mixed_nmr_spectrum(sequences, ratio, nmr_spectrum_generator)
                ms_ds[i, :] = get_mixed_ms_spectrum(sequences, ratio, ms_spectrum_generator)
                sequence_ds[i] = [seq.encode('utf-8') for seq in sequences]
                ratio_ds[i] = ratio
                lambda_ds[i] = lambd

def get_mixed_uv_vis_spectrum(sequences, ratios, uv_vis_spectrum_generator):
    """Helper function to get the mixed UV-Vis spectrum of a mixture."""
    uv_vis_spectrum = np.zeros(uv_vis_spectrum_generator.points)
    for sequence, ratio in zip(sequences, ratios):
        uv_vis_spectrum += ratio * uv_vis_spectrum_generator.getSpectrum(sequence)
    return uv_vis_spectrum

def get_mixed_nmr_spectrum(sequences, ratios, nmr_spectrum_generator):
    """Helper function to get the mixed NMR spectrum of a mixture."""
    nmr_spectrum = np.zeros(nmr_spectrum_generator.points)

    trimers_mixed = {}
    endgroups_mixed = {}

    for sequence, ratio in zip(sequences, ratios):
        # get counts of each trimer in the sequence
        trimers = nmr_spectrum_generator._countTrimers(sequence)

        # add endgroups to the counts
        endgroups = nmr_spectrum_generator._countEndgroups(sequence)

        # add to mixed trimers and endgroups
        for trimer in trimers:
            if trimer not in trimers_mixed:
                trimers_mixed[trimer] = 0
            trimers_mixed[trimer] += ratio * trimers[trimer]

        for endgroup in endgroups:
            if endgroup not in endgroups_mixed:
                endgroups_mixed[endgroup] = 0
            endgroups_mixed[endgroup] += ratio * endgroups[endgroup]

    # generate the aggregate shifts and intensities for the mixtures
    spectrum = nmr_spectrum_generator._generateSpectrum(trimers_mixed, endgroups_mixed)

    # generate NMR spectrum for mixtures
    nmr_spectrum = nmr_spectrum_generator._generateLorentzian(spectrum, normalize=True)

    return nmr_spectrum

def get_mixed_ms_spectrum(sequences, ratios, ms_spectrum_generator):
    """Helper function to get the mixed MS spectrum of a mixture."""
    ms_spectrum = np.zeros(len(ms_spectrum_generator.x))

    for sequence, ratio in zip(sequences, ratios):
        ms_spectrum_i = ms_spectrum_generator.getSpectrum(sequence)
        ms_spectrum += ratio * ms_spectrum_i #/ sum(ms_spectrum_i)  # normalize each spectrum before adding

    # max normalize final spectrum
    ms_spectrum = ms_spectrum / max(ms_spectrum)

    return ms_spectrum

def plot_mixture(sequences, ratios, normalize=True):

    uv_vis_spectrum_generator = UV_Vis(frenkel_params, uv_vis_plot_params)
    nmr_spectrum_generator = NMR(trimer_file, dimer_file, nmr_plot_params)
    ms_spectrum_generator = MS(ms_parameters, ms_plot_parameters, ms_noise_parameters)

    uv_vis_spectrum = get_mixed_uv_vis_spectrum(sequences, ratios, uv_vis_spectrum_generator)
    nmr_spectrum = get_mixed_nmr_spectrum(sequences, ratios, nmr_spectrum_generator)
    ms_spectrum = get_mixed_ms_spectrum(sequences, ratios, ms_spectrum_generator)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for seq in sequences:
        axs[0].plot(uv_vis_spectrum_generator.x, uv_vis_spectrum_generator.getSpectrum(seq, normalize=normalize), label=seq, alpha=0.5)
    if normalize:  
        uv_vis_spectrum = uv_vis_spectrum / np.max(uv_vis_spectrum)
    axs[0].plot(uv_vis_spectrum_generator.x, uv_vis_spectrum, label='Mixed', color='black')
    axs[0].set_xlabel('Energy (eV)')
    axs[0].set_ylabel('Absorbance')
    axs[0].set_title('UV-Vis Spectrum')
    axs[0].legend(loc='upper right')

    for seq in sequences:
        axs[1].plot(nmr_spectrum_generator.x, nmr_spectrum_generator.getSpectrum(seq), label=seq, alpha=0.5)
    axs[1].plot(nmr_spectrum_generator.x, nmr_spectrum, label='Mixed', color='black')
    axs[1].invert_xaxis()
    axs[1].set_xlabel('Chemical Shift (ppm)')
    axs[1].set_ylabel('Intensity')
    axs[1].set_title('NMR Spectrum')

    for seq in sequences:
        axs[2].bar(ms_spectrum_generator.x, ms_spectrum_generator.getSpectrum(seq), width=2*ms_plot_parameters.binWidth, label=seq, alpha=0.5)
    axs[2].bar(ms_spectrum_generator.x, ms_spectrum, width=2*ms_plot_parameters.binWidth, label='Mixed', color='black')
    axs[2].set_xlabel('Mass-to-Charge Ratio (m/z)')
    axs[2].set_ylabel('Intensity')
    axs[2].set_title('Mass Spectrum')

    plt.tight_layout()
    
    return fig

if __name__ == '__main__':
    #main()
    normalize=True
    for lamb in lambdas:
        mixtures_file = f'mixtures_L10_lamb{lamb}.csv'  # file with sequences and lambdas
        reader = csv.reader(open(mixtures_file, 'r', encoding="utf-8-sig"))

        folder = f'./Output/datasets/mixtures/norm_{normalize}/mixture_{lamb}/'
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        for i in range(10):
            
            row = next(reader)  
            sequences = ast.literal_eval(row[0])
            ratio = ast.literal_eval(row[1])

            fig = plot_mixture(sequences, ratio, normalize=normalize)
            plt.savefig(folder + f'{i}.png')
            plt.close()

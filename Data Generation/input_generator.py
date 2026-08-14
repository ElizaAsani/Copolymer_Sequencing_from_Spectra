"""
Copolymer Input Generation
"""

import csv
import h5py
import numpy as np
from UV_Vis import *
from NMR import *
from MS import *

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
    std_dev = 0.2,                  # standard deviation (0.1)
    energyRange = [0.4, 6.5]       # energy range to calculate absorption over ([0.4, 6.5]))
)

#------------NMR Parameters-----------#
trimer_file = 'HNMR_trimers.csv'  # file with trimer data
dimer_file = 'HNMR_dimers.csv'    # file with dimer data

nmr_plot_params = NMRPlotParameters(
    points = 500,                   # number of points on the curve to calculate (500)
    half_width = 0.4,               # half-width of the Lorentzian peak (0.1)
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
        extra_peaks = 5,
        width = 182,
        weight = 0.2,
    )

SIM_UV_VIS = True
SIM_NMR = False
SIM_MS = False

def main():

    #date = '05_20_26'
    filename = f'./Output/datasets/uv_vis/std0.2.h5'

    sequence_file = 'seq_3-20.csv'  # file with sequences and lambdas
    reader = csv.reader(open(sequence_file, 'r'))
    sequences, lambdas = zip(*[(row[0], float(row[1])) for row in reader])

    num_sequences = len(sequences)      
   
    # initialize spectrum generators
    if SIM_UV_VIS:
        uv_vis_spectrum_generator = UV_Vis(frenkel_params, uv_vis_plot_params)
    if SIM_NMR:
        nmr_spectrum_generator = NMR(trimer_file, dimer_file, nmr_plot_params)
    if SIM_MS:
        ms_spectrum_generator = MS(ms_parameters, ms_plot_parameters, ms_noise_parameters)
    
    # open and write to file
    with h5py.File(filename, 'w') as f:

        if SIM_UV_VIS:        
            uv_vis_ds = f.create_dataset(
                'uv_vis',
                shape=(num_sequences, uv_vis_plot_params.points),
                dtype=np.float32,
                chunks=(1, uv_vis_plot_params.points)
            )
            uv_vis_ds.attrs['points'] = uv_vis_plot_params.points
            uv_vis_ds.attrs['std_dev'] = uv_vis_plot_params.std_dev
            uv_vis_ds.attrs['energy_range'] = uv_vis_spectrum_generator.energyRange
            if uv_vis_plot_params.wavelengthRange is not None:
                uv_vis_ds.attrs['wavelength_range'] = uv_vis_plot_params.wavelengthRange
            if uv_vis_spectrum_generator.peaks is not None:
                uv_vis_ds.attrs['peaks'] = uv_vis_spectrum_generator.peaks
        
        if SIM_NMR:
            nmr_ds = f.create_dataset(
                'nmr',
                shape=(num_sequences, nmr_plot_params.points),
                dtype=np.float32,       
                chunks=(1, nmr_plot_params.points)
            )
            nmr_ds.attrs['points'] = nmr_plot_params.points
            nmr_ds.attrs['shift_range'] = nmr_spectrum_generator.shift_range
            nmr_ds.attrs['half_width'] = nmr_plot_params.half_width

        if SIM_MS:
            ms_ds = f.create_dataset(
                'ms',
                shape=(num_sequences, len(ms_spectrum_generator.x)),
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
            data = sequences,
            dtype = h5py.string_dtype(encoding='utf-8'),
            chunks=(1,)
        )   
        sequence_ds.attrs['monomers'] = monomers
        sequence_ds.attrs['max_length'] = max(len(seq) for seq in sequences)

        lambda_ds = f.create_dataset(
            'lambda',
            data = lambdas,
            dtype = np.float32,
            chunks=(1,)
        )
        
        for i in range(num_sequences):
            # get sequence and lambda
            sequence = sequences[i]
            
            # generate and store spectra
            if SIM_UV_VIS:
                uv_vis_ds[i, :] = uv_vis_spectrum_generator.getSpectrum(sequence)
            if SIM_NMR:
                nmr_ds[i, :] = nmr_spectrum_generator.getSpectrum(sequence)
            if SIM_MS:
                ms_ds[i, :] = ms_spectrum_generator.getSpectrum(sequence)

            if (i % 1000 == 0):
                print(i)

if __name__ == '__main__':
    main()

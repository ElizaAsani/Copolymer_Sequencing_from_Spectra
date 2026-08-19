"""
Copolymer Input Generation
"""

import argparse
import csv
import h5py
import numpy as np

from UV_Vis import *
from NMR import *
from MS import *
from spectra_config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Generate copolymer spectra for a given set of sequences and lambdas.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def run(cfg):
    """Generate dataset of copolymer spectra for a given configuration."""

    reader = csv.reader(open(cfg.sequence_file, 'r'))
    sequences, lambdas = zip(*[(row[0], float(row[1])) for row in reader])

    num_sequences = len(sequences)      
   
    # initialize spectrum generators
    if cfg.simulate_uv_vis:
        uv_vis_spectrum_generator = UV_Vis(cfg.frenkel_params, cfg.uv_vis_plot_params)
    if cfg.simulate_nmr:
        nmr_spectrum_generator = NMR(cfg.trimer_file, cfg.dimer_file, cfg.nmr_plot_params)
    if cfg.simulate_ms:
        ms_spectrum_generator = MS(cfg.ms_params, cfg.ms_plot_params, cfg.ms_noise_params)
    
    # open and write to file
    with h5py.File(cfg.output_file, 'w') as f:

        if cfg.simulate_uv_vis:        
            uv_vis_ds = f.create_dataset(
                'uv_vis',
                shape=(num_sequences, uv_vis_spectrum_generator.points),
                dtype=np.float32,
                chunks=(1, uv_vis_spectrum_generator.points)
            )
            uv_vis_ds.attrs['points'] = uv_vis_spectrum_generator.points
            uv_vis_ds.attrs['std_dev'] = uv_vis_spectrum_generator.std_dev
            uv_vis_ds.attrs['energy_range'] = uv_vis_spectrum_generator.energy_range
            if uv_vis_spectrum_generator.wavelength_range is not None:
                uv_vis_ds.attrs['wavelength_range'] = uv_vis_spectrum_generator.wavelength_range
            if uv_vis_spectrum_generator.peaks is not None:
                uv_vis_ds.attrs['peaks'] = uv_vis_spectrum_generator.peaks
        
        if cfg.simulate_nmr:
            nmr_ds = f.create_dataset(
                'nmr',
                shape=(num_sequences, nmr_spectrum_generator.points),
                dtype=np.float32,       
                chunks=(1, nmr_spectrum_generator.points)
            )
            nmr_ds.attrs['points'] = nmr_spectrum_generator.points
            nmr_ds.attrs['shift_range'] = nmr_spectrum_generator.shift_range
            nmr_ds.attrs['half_width'] = nmr_spectrum_generator.half_width

        if cfg.simulate_ms:
            ms_ds = f.create_dataset(
                'ms',
                shape=(num_sequences, len(ms_spectrum_generator.x)),
                dtype=np.float32,
                chunks=(1, len(ms_spectrum_generator.x))
            )
            ms_ds.attrs['points'] = len(ms_spectrum_generator.x)
            ms_ds.attrs['mass_range'] = ms_spectrum_generator.mass_range
            ms_ds.attrs['bin_width'] = ms_spectrum_generator.bin_width
            ms_ds.attrs['dropout'] = ms_spectrum_generator.dropout
            ms_ds.attrs['extra_peaks'] = ms_spectrum_generator.extra_peaks
            ms_ds.attrs['noise_width'] = ms_spectrum_generator.noise_width
            ms_ds.attrs['peak_weight'] = ms_spectrum_generator.peak_weight

        sequence_ds = f.create_dataset(
            'sequence',
            data = sequences,
            dtype = h5py.string_dtype(encoding='utf-8'),
            chunks=(1,)
        )   
        sequence_ds.attrs['monomers'] = cfg.monomers
        lengths = [len(seq) for seq in sequences]
        sequence_ds.attrs['min_length'] = min(lengths)
        sequence_ds.attrs['max_length'] = max(lengths)

        f.create_dataset(
            'lambda',
            data = lambdas,
            dtype = np.float32,
            chunks=(1,)
        )
        
        for i in range(num_sequences):
            # get sequence and lambda
            sequence = sequences[i]
            
            # generate and store spectra
            if cfg.simulate_uv_vis:
                uv_vis_ds[i, :] = uv_vis_spectrum_generator.getSpectrum(sequence)
            if cfg.simulate_nmr:
                nmr_ds[i, :] = nmr_spectrum_generator.getSpectrum(sequence)
            if cfg.simulate_ms:
                ms_ds[i, :] = ms_spectrum_generator.getSpectrum(sequence)

            if (i % 1000 == 0):
                print(i)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)

if __name__ == '__main__':
    main()

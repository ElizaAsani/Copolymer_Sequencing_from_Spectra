"""
Copolymer Spectra Generation
"""

import argparse
import ast
import os
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
    if cfg.mode == "single":
        run_single(cfg)
    elif cfg.mode == "mixtures":
        run_mixtures(cfg)

def run_single(cfg):
    """mode: single - one sequence per spectrum"""

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

def run_mixtures(cfg):
    """mode: mixtures - multiple sequences per spectrum"""
   
    # initialize spectrum generators
    if cfg.simulate_uv_vis:
        uv_vis_spectrum_generator = UV_Vis(cfg.frenkel_params, cfg.uv_vis_plot_params)
    if cfg.simulate_nmr:
        nmr_spectrum_generator = NMR(cfg.trimer_file, cfg.dimer_file, cfg.nmr_plot_params)
    if cfg.simulate_ms:
        ms_spectrum_generator = MS(cfg.ms_params, cfg.ms_plot_params, cfg.ms_noise_params)

    for lamb in cfg.mixtures_lambdas:
        mixtures_file = os.path.join(cfg.output_dir, f"mixtures_lamb{lamb}.csv")
        output_file = os.path.join(cfg.output_dir, f"mixtures_lamb{lamb}.h5")

        reader = csv.reader(open(mixtures_file, 'r', encoding="utf-8-sig"))
        mixtures = []
        ratios = []

        for row in reader:
            sequences = ast.literal_eval(row[0])
            ratio = ast.literal_eval(row[1])

            mixtures.append(sequences)
            ratios.append(ratio)

        num_mixtures = len(mixtures)
        lambdas = [lamb] * num_mixtures
    
        # open and write to file
        with h5py.File(output_file, 'w') as f:

            if cfg.simulate_uv_vis:        
                uv_vis_ds = f.create_dataset(
                    'uv_vis',
                    shape=(num_mixtures, uv_vis_spectrum_generator.points),
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
                    shape=(num_mixtures, nmr_spectrum_generator.points),
                    dtype=np.float32,       
                    chunks=(1, nmr_spectrum_generator.points)
                )
                nmr_ds.attrs['points'] = nmr_spectrum_generator.points
                nmr_ds.attrs['shift_range'] = nmr_spectrum_generator.shift_range
                nmr_ds.attrs['half_width'] = nmr_spectrum_generator.half_width

            if cfg.simulate_ms:
                ms_ds = f.create_dataset(
                    'ms',
                    shape=(num_mixtures, len(ms_spectrum_generator.x)),
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
                shape=(num_mixtures,),
                dtype=h5py.vlen_dtype(h5py.string_dtype(encoding='utf-8')),
                chunks=(1,)
            )
            sequence_ds.attrs['monomers'] = cfg.monomers
            sequence_ds.attrs['max_length'] = 20            # hard-coded to correspond to transformer model max length

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
                # get sequences and ratios
                sequences = mixtures[i]
                ratio = ratios[i]
                lamb = lambdas[i]
                
                # generate and store spectra
                if cfg.simulate_uv_vis:
                    uv_vis_ds[i, :] = get_mixed_uv_vis_spectrum(sequences, ratio, uv_vis_spectrum_generator)
                if cfg.simulate_nmr:
                    nmr_ds[i, :] = get_mixed_nmr_spectrum(sequences, ratio, nmr_spectrum_generator)
                if cfg.simulate_ms:
                    ms_ds[i, :] = get_mixed_ms_spectrum(sequences, ratio, ms_spectrum_generator)

                # store sequences, ratios, and lambda
                sequence_ds[i] = sequences
                ratio_ds[i] = ratio
                lambda_ds[i] = lamb

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

def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)

if __name__ == '__main__':
    main()

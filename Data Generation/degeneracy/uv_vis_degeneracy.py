"""
Mass spectra degeneracy analysis
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from UV_Vis import UV_Vis
from spectra_config import load_config
from degeneracy_functions import (
    threshold_match_degeneracy, 
    all_lengths_degeneracy, 
    dataset_degeneracies, 
    degeneracy_aware_reconstruction)

MIN_LENGTH, MAX_LENGTH = 3, 20
SEQUENCE_DIR = os.path.join('..', 'all')
SEQUENCE_FILE = os.path.join('..', f'seq_{MIN_LENGTH}-{MAX_LENGTH}.csv')
OUT_FOLDER = 'uv_vis'
MODEL_OUTPUT_FOLDER = os.path.join('..', '..', 'Multispectra-to-sequence Transformer', 'Output', 'uv_vis', 'NOISE0', 'errors')

SPECTRA_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'spectra', 'NOISE0.yaml')
cfg = load_config(SPECTRA_CONFIG)

uv_vis_spectrum_generator = UV_Vis(cfg.frenkel_params, cfg.uv_vis_plot_params)

X_RES = 0.01 if uv_vis_spectrum_generator.unit == 'energy' else 1 # threshold for considering two peaks as the same in energy/wavelength
INTENSITY_RES = 0.01  # threshold for considering a peak as non-zero

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def getDiscreteUVVIS(sequence, max_length=None):
    """Get discrete mass spectrum for a given sequence.
    Returns a tuple of rounded absorption energies and intensities.
    """
    absorption = uv_vis_spectrum_generator.getAbsorption(sequence)
    if uv_vis_spectrum_generator.unit == 'energy':
        x = np.array(absorption['energy'])
    elif uv_vis_spectrum_generator.unit == 'wavelength': 
        x = np.array(absorption['wavelength'])

    intensity = np.array(absorption['intensity'])
    intensity = intensity / intensity.max()  # normalize intensity to max value

    # drop zero-intensity values
    non_zero = intensity >= INTENSITY_RES
    x = x[non_zero]
    intensity = intensity[non_zero]

    # pad to max length
    if max_length is not None:
        x = np.pad(x, (0, max_length - len(x)), constant_values=0)
        intensity = np.pad(intensity, (0, max_length - len(intensity)), constant_values=0)

    return x, intensity

def isIndistinguishableUVVIS(seq1, seq2):
    """Check if two sequences are indistinguishable based on their UV-Vis spectra."""

    max_length = max(len(seq1), len(seq2))

    x1, intensity1 = getDiscreteUVVIS(seq1, max_length)
    x2, intensity2 = getDiscreteUVVIS(seq2, max_length)

    return (np.all(np.abs(x1 - x2) < X_RES) and
            np.all(np.abs(intensity1 - intensity2) < INTENSITY_RES))

def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    
    all_lengths_degeneracy(getDiscreteUVVIS, 
                           lambda seqs, getDiscrete, out_file: threshold_match_degeneracy(seqs, getDiscrete, 
                                                                                          x_res=X_RES, intensity_res=INTENSITY_RES, 
                                                                                          out_file=out_file, device=device),
                            OUT_FOLDER, 20, MAX_LENGTH, SEQUENCE_DIR)

    dataset_degeneracies(OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_FILE)

    degeneracy_aware_reconstruction(isIndistinguishableUVVIS, MODEL_OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
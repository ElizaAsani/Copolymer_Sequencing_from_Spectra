"""
Mass spectra degeneracy analysis
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from MS import MS
from spectra_config import load_config
from degeneracy_functions import (
    exact_match_degeneracy, 
    all_lengths_degeneracy, 
    dataset_degeneracies, 
    degeneracy_aware_reconstruction)

MIN_LENGTH, MAX_LENGTH = 3, 20
SEQUENCE_DIR = os.path.join('..', 'all')
SEQUENCE_FILE = os.path.join('..', f'seq_{MIN_LENGTH}-{MAX_LENGTH}.csv')
OUT_FOLDER = 'ms'
MODEL_OUTPUT_FOLDER = os.path.join('..', '..', 'Multispectra-to-sequence Transformer', 'Output', 'ms', 'NOISE0', 'errors')

SPECTRA_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'spectra', 'NOISE0.yaml')
cfg = load_config(SPECTRA_CONFIG)

ms_spectrum_generator = MS(cfg.ms_params, cfg.ms_plot_params, cfg.ms_noise_params)

def getDiscreteMS(sequence):
    """Get discrete mass spectrum for a given sequence."""
    return tuple(ms_spectrum_generator.generateMassSpectrum(sequence).keys())

def isIndistinguishableMS(seq1, seq2):
    """Check if two sequences are indistinguishable based on their mass spectra."""
    spectrum1 = getDiscreteMS(seq1)
    spectrum2 = getDiscreteMS(seq2)
    return np.array_equal(spectrum1, spectrum2)

def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    all_lengths_degeneracy(getDiscreteMS, exact_match_degeneracy, OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_DIR)

    dataset_degeneracies(OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_FILE)

    degeneracy_aware_reconstruction(isIndistinguishableMS, MODEL_OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
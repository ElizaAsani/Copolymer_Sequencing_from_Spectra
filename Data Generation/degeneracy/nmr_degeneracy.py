"""
Mass spectra degeneracy analysis
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from NMR import NMR
from spectra_config import load_config
from degeneracy_functions import (
    exact_match_degeneracy, 
    all_lengths_degeneracy, 
    dataset_degeneracies, 
    degeneracy_aware_reconstruction)

MIN_LENGTH, MAX_LENGTH = 3, 20
SEQUENCE_DIR = os.path.join('..', 'all')
SEQUENCE_FILE = os.path.join('..', f'seq_{MIN_LENGTH}-{MAX_LENGTH}.csv')
OUT_FOLDER = 'nmr'
MODEL_OUTPUT_FOLDER = os.path.join('..', '..', 'Multispectra-to-sequence Transformer', 'Output', 'nmr', 'NOISE0', 'errors')

SPECTRA_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'spectra', 'NOISE0.yaml')
cfg = load_config(SPECTRA_CONFIG)

TRIMER_FILE = os.path.join(os.path.dirname(__file__), '..', cfg.trimer_file)
DIMER_FILE = os.path.join(os.path.dirname(__file__), '..', cfg.dimer_file)

nmr_spectrum_generator = NMR(TRIMER_FILE, DIMER_FILE, cfg.nmr_plot_params)

def getDiscreteNMR(sequence):
    """Get discrete NMR spectrum for a given sequence."""
    trimers = nmr_spectrum_generator._countTrimers(sequence)
    endgroups = nmr_spectrum_generator._countEndgroups(sequence)
    subgroups = trimers | endgroups
    return tuple(subgroups.values())

def isIndistinguishableNMR(seq1, seq2):
    """Check if two sequences are indistinguishable based on their NMR spectra."""
    spectrum1 = getDiscreteNMR(seq1)
    spectrum2 = getDiscreteNMR(seq2)
    return np.array_equal(spectrum1, spectrum2)

def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    all_lengths_degeneracy(getDiscreteNMR, exact_match_degeneracy, OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_DIR)

    dataset_degeneracies(OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_FILE)

    degeneracy_aware_reconstruction(isIndistinguishableNMR, MODEL_OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
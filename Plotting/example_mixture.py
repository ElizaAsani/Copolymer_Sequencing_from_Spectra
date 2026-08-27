"""
Example single-sequence spectra plots.
"""

import argparse
import ast
import csv

import os
import sys

data_gen_dir = os.path.join(os.path.dirname(__file__), '..', 'Data Generation')
sys.path.append(data_gen_dir)

import numpy as np
import matplotlib.pyplot as plt

from UV_Vis import UV_Vis
from NMR import NMR
from MS import MS
from spectra_config import load_config
from spectra_generator import get_mixed_uv_vis_spectrum, get_mixed_nmr_spectrum, get_mixed_ms_spectrum

plt.style.use('figure.mplstyle')

def plot_mixture(sequences, ratios, cfg):

    uv_vis_spectrum_generator = UV_Vis(cfg.frenkel_params, cfg.uv_vis_plot_params)
    trimer_file = os.path.join(data_gen_dir, cfg.trimer_file)
    dimer_file = os.path.join(data_gen_dir, cfg.dimer_file)
    nmr_spectrum_generator = NMR(trimer_file, dimer_file, cfg.nmr_plot_params)
    ms_spectrum_generator = MS(cfg.ms_params, cfg.ms_plot_params, cfg.ms_noise_params)

    uv_vis_spectrum = get_mixed_uv_vis_spectrum(sequences, ratios, uv_vis_spectrum_generator)
    nmr_spectrum = get_mixed_nmr_spectrum(sequences, ratios, nmr_spectrum_generator)
    ms_spectrum = get_mixed_ms_spectrum(sequences, ratios, ms_spectrum_generator)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # ---- UV-Vis ---- #
    for seq in sequences:
        axs[0].plot(uv_vis_spectrum_generator.x, uv_vis_spectrum_generator.getSpectrum(seq, normalize=True), label=seq, alpha=0.5)

    uv_vis_spectrum = uv_vis_spectrum / max(uv_vis_spectrum) # normalize mixture UV-Vis spectrum

    axs[0].plot(uv_vis_spectrum_generator.x, uv_vis_spectrum, label='Mixed', color='black')
    axs[0].set_xlabel('Energy (eV)')
    axs[0].set_ylabel('Absorbance')
    axs[0].set_title('UV-Vis Spectrum')
    axs[0].legend(loc='upper right')

    # ---- NMR ---- #
    for seq in sequences:
        axs[1].plot(nmr_spectrum_generator.x, nmr_spectrum_generator.getSpectrum(seq), label=seq, alpha=0.5)

    axs[1].plot(nmr_spectrum_generator.x, nmr_spectrum, label='Mixed', color='black')
    axs[1].invert_xaxis()
    axs[1].set_xlabel('Chemical Shift (ppm)')
    axs[1].set_ylabel('Intensity')
    axs[1].set_title('NMR Spectrum')

    # ---- MS ---- #
    for seq in sequences:
        axs[2].bar(ms_spectrum_generator.x, ms_spectrum_generator.getSpectrum(seq), width=2 * cfg.ms_plot_params.bin_width, label=seq, alpha=0.5)

    axs[2].bar(ms_spectrum_generator.x, ms_spectrum, width=2 * cfg.ms_plot_params.bin_width, label='Mixed', color='black')
    axs[2].set_xlabel('Mass-to-Charge Ratio (m/z)')
    axs[2].set_ylabel('Intensity')
    axs[2].set_title('Mass Spectrum')

    plt.tight_layout()

    return fig

def save_mixture_examples(cfg, output_dir, n_per_lambda):
    """Plots n_per_lambda UV-Vis/NMR/MS mixtures spectrum plots for each
    lambda in cfg.mixtures_lambdas, and saves them to output_dir. Mixture
    sequence are read from the generated mixtures CSV files."""

    os.makedirs(output_dir, exist_ok=True)

    for lamb in cfg.mixtures_lambdas:

        lambda_dir = os.path.join(output_dir, f"lambd{lamb}")
        os.makedirs(lambda_dir, exist_ok=True)

        mixtures_file = os.path.join(data_gen_dir, cfg.output_dir, f"mixtures_lamb{lamb}.csv")
        reader = csv.reader(open(mixtures_file, 'r', encoding="utf-8-sig"))

        for i in range(n_per_lambda):
            row = next(reader)
            sequences = ast.literal_eval(row[0])
            ratios = ast.literal_eval(row[1])

            fig = plot_mixture(sequences, ratios, cfg)
            fig.savefig(os.path.join(lambda_dir, f"{i}.svg"))
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example spectra for a single copolymer sequence.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file, relative to ../Data Generation/configs/spectra/.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files, relative to ../examples/.")
    parser.add_argument("--n", type=int, default=5, help="Number of example mixture plots to save for each lambda.")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', 'Data Generation', 'configs', 'spectra', args.config)
    output_dir = os.path.join("examples", args.output_dir)
    
    cfg = load_config(config_path)
    save_mixture_examples(cfg, output_dir=output_dir, n_per_lambda=args.n)

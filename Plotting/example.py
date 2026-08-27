"""
Example single-sequence spectra plots.
"""

import argparse
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

np.set_printoptions(suppress=True)
plt.style.use('figure.mplstyle')

palette = ['#6A4E5A','#A72420', '#909033']

colors = {
    "NMR": palette[0],          
    "MS": palette[1],            
    "UV-Vis": palette[2],                  
}

def plot_eigenvalue(eigenvectors, eigenvalues, sequence):
    """Plot the eigenvalues and eigenvectors of the Frenkel Hamiltonian for a given sequence."""
    # sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # plot eigenvectors
    fig, axs = plt.subplots(len(eigenvalues), 1, figsize=(4, 1.2 * len(eigenvalues)), constrained_layout=True)
    for i in range(len(eigenvalues)):
        ax = axs[i]
        x = np.arange(len(sequence))
        y = eigenvectors[:, i]
        pos_idx = np.where(y >= 0)[0]
        neg_idx = np.where(y < 0)[0]
        if len(pos_idx) > 0:
            ax.stem(x[pos_idx], y[pos_idx], linefmt=colors['UV-Vis'], markerfmt='^', basefmt=' ')
        if len(neg_idx) > 0:    
            ax.stem(x[neg_idx], y[neg_idx], linefmt=colors['UV-Vis'], markerfmt='v', basefmt=' ')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
        ax.set_xlabel('Monomer')
        ax.set_xticks(x)
        ax.set_xticklabels(list(sequence))
        ax.set_ylabel(f'$\\phi_{{{i+1}}}$')
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'$E_{{{i+1}}}$: {eigenvalues[i]:.2f} eV')
    return fig

def save_single_sequence_example(cfg, sequence, output_dir):
    """Writes UV-Vis/NMR/MS .txt files and spectrum plots for a
    single copolymer sequence into output_dir."""

    os.makedirs(output_dir, exist_ok=True)

    uv_vis_spectrum_generator = UV_Vis(cfg.frenkel_params, cfg.uv_vis_plot_params)
    trimer_file = os.path.join(data_gen_dir, cfg.trimer_file)
    dimer_file = os.path.join(data_gen_dir, cfg.dimer_file)
    nmr_spectrum_generator = NMR(trimer_file, dimer_file, cfg.nmr_plot_params)
    ms_spectrum_generator = MS(cfg.ms_params, cfg.ms_plot_params, cfg.ms_noise_params)

    # ---- UV-Vis ---- #
    with open(os.path.join(output_dir, 'uv_vis.txt'), 'w') as f:
        f.write(f"UV-Vis Spectrum for sequence: {sequence}\n")
        f.write(f"Energy Range: {uv_vis_spectrum_generator.energy_range}\n")
        f.write(f"Points: {uv_vis_spectrum_generator.points}\n")
        f.write(f"Standard Deviation: {uv_vis_spectrum_generator.std_dev}\n")
        if uv_vis_spectrum_generator.peaks is not None:
            f.write(f"Peaks: {uv_vis_spectrum_generator.peaks}\n\n")

        f.write(f"Monomer Energies: {uv_vis_spectrum_generator.eps}\n")
        f.write(f"Monomer Dipole Moments: {uv_vis_spectrum_generator.mu}\n")
        f.write(f"Dipole-Dipole Coupling Coefficient: {uv_vis_spectrum_generator.J_DD}\n")
        f.write(f"Superexchange Coupling Coefficient: {uv_vis_spectrum_generator.J_SE}\n")

        H = uv_vis_spectrum_generator._generateHamiltonian(sequence)
        f.write(f"Hamiltonian Matrix:\n{np.round(H, 3)}\n\n")

        eigenvalues, eigenvectors = np.linalg.eig(H)
        f.write(f"Eigenvalues (Exciton Energies):\n{np.round(eigenvalues, 3)}\n\n")
        f.write(f"Eigenvectors (Exciton States):\n{np.round(eigenvectors, 3)}\n\n")
        eigen_fig = plot_eigenvalue(eigenvectors, eigenvalues, sequence)
        eigen_fig.savefig(os.path.join(output_dir, 'eigenvectors.svg'))
        plt.close(eigen_fig)

        absorption = uv_vis_spectrum_generator._generateAbsorption(eigenvalues, eigenvectors, sequence)
        f.write(f"Absorption Spectrum:\n{absorption}\n\n")

        truncated_absorption = uv_vis_spectrum_generator.getAbsorption(sequence)
        f.write(f"Truncated Absorption Spectrum:\n{truncated_absorption}\n\n")

    x = uv_vis_spectrum_generator.x_energy
    y = uv_vis_spectrum_generator.getSpectrum(sequence, normalize=True)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(x, y, color=colors["UV-Vis"], label='UV-Vis Spectrum')
    ax.set_xlabel('Energy [eV]')    
    ax.set_ylabel('Intensity')
    ax.set_xlim(uv_vis_spectrum_generator.energy_range)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid()
    ax.set_axisbelow('on')
    fig.savefig(os.path.join(output_dir, 'uv_vis_spectrum.svg'))
    plt.close(fig)

    # ---- NMR ---- #
    with open(os.path.join(output_dir, 'nmr.txt'), 'w') as f:
        f.write(f"NMR Spectrum for sequence: {sequence}\n")
        f.write(f"Chemical Shift Range: {nmr_spectrum_generator.shift_range}\n")
        f.write(f"Points: {nmr_spectrum_generator.points}\n")
        f.write(f"Half Width: {nmr_spectrum_generator.half_width}\n\n")

        trimers = nmr_spectrum_generator._countTrimers(sequence)
        f.write(f"Trimers Count:\n{trimers}\n\n")

        endgroups = nmr_spectrum_generator._countEndgroups(sequence)
        f.write(f"Endgroups Count:\n{endgroups}\n\n")

        spectrum = nmr_spectrum_generator._generateSpectrum(trimers, endgroups)
        f.write(f"NMR Spectrum:\n{spectrum}\n\n")

    x = nmr_spectrum_generator.x
    y = nmr_spectrum_generator.getSpectrum(sequence)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(x, y, color=colors["NMR"], label='NMR Spectrum')
    ax.set_xlabel('Chemical Shift [ppm]')
    ax.set_ylabel('Intensity')
    ax.set_xlim(nmr_spectrum_generator.shift_range)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid()
    ax.set_axisbelow('on')
    fig.savefig(os.path.join(output_dir, 'nmr_spectrum.svg'))
    plt.close(fig)

    # ---- MS ---- #
    with open(os.path.join(output_dir, 'ms.txt'), 'w') as f:
        f.write(f"Mass Spectrum for sequence: {sequence}\n")
        f.write(f"Mass Range: {ms_spectrum_generator.mass_range}\n")
        f.write(f"Bin Width: {ms_spectrum_generator.bin_width}\n\n")

        f.write(f"Dropout: {ms_spectrum_generator.dropout}\n")
        f.write(f"Peak Weight: {ms_spectrum_generator.peak_weight}\n")
        f.write(f"Extra Peaks: {ms_spectrum_generator.extra_peaks}\n")
        f.write(f"Noise Width: {ms_spectrum_generator.noise_width}\n\n")

        fragments = ms_spectrum_generator._getFragments(sequence)
        f.write(f"{'Fragment':<12}{'Intensity':>12}\n")
        f.write(f"{'-' * 12}{'-' * 12}\n")
        for fragment, intensity in fragments.items():
            f.write(f"{str(fragment):<12}{np.round(intensity, 3):>12.3f}\n")

        mass_spectrum = ms_spectrum_generator.generateMassSpectrum(sequence)
        f.write("\n")
        f.write(f"{'Mass':<12}{'Intensity':>12}\n")
        f.write(f"{'-' * 12}{'-' * 12}\n")
        for mass, intensity in mass_spectrum.items():
            f.write(f"{str(mass):<12}{np.round(intensity, 3):>12.3f}\n")

    x = ms_spectrum_generator.x
    y = ms_spectrum_generator.getSpectrum(sequence)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.bar(x, y, width=ms_spectrum_generator.bin_width, color=colors["MS"], label='Mass Spectrum')
    ax.set_xlabel('Mass [m/z]')
    ax.set_ylabel('Intensity')
    ax.set_xlim(ms_spectrum_generator.mass_range[0] - ms_spectrum_generator.bin_width, 
                max(mass_spectrum.keys()) + ms_spectrum_generator.bin_width)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid(axis='y')
    ax.set_axisbelow('on')
    fig.savefig(os.path.join(output_dir, 'ms_spectrum.svg'))
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example spectra for a single copolymer sequence.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file, relative to ../Data Generation/configs/spectra/.")
    parser.add_argument("--sequence", type=str, default='ADDAADAAAD', help="Copolymer sequence to generate spectra for.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files, relative to ../examples/.")
    args = parser.parse_args()


    config_path = os.path.join(os.path.dirname(__file__), '..', 'Data Generation', 'configs', 'spectra', args.config)
    output_dir = os.path.join("examples", args.output_dir)
    
    cfg = load_config(config_path)
    save_single_sequence_example(cfg, args.sequence, output_dir)

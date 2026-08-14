"""
Copolymer Input Generation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from UV_Vis import *
from NMR import *
from MS import *

np.set_printoptions(suppress=True)
plt.style.use('rsc.mplstyle')
palette = ['#6A4E5A','#A72420', '#909033']
colors = {
    "NMR": palette[0],          
    "MS": palette[1],            
    "UV-Vis": palette[2]          
}

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
        width = 182,
        weight = 1,
    )

def plot_eigenvalue(eigenvectors, eigenvalues, sequence):
    """Plot the eigenvalues and eigenvectors of a given sequence."""
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

sequence = 'ADDAADAAAD'
path = rf"./Output/Eliza - Project Plots/{sequence}/NOISE0/"
if not os.path.exists(path):
    os.makedirs(path)

def main():

    # initialize spectrum generators
    uv_vis_spectrum_generator = UV_Vis(frenkel_params, uv_vis_plot_params)
    nmr_spectrum_generator = NMR(trimer_file, dimer_file, nmr_plot_params)
    ms_spectrum_generator = MS(ms_parameters, ms_plot_parameters, ms_noise_parameters)
   
    # generate and plot spectra
    uv_vis_file = 'uv_vis.txt'
    with open(os.path.join(path, uv_vis_file), 'w') as f:
        f.write(f"UV-Vis Spectrum for sequence: {sequence}\n")
        f.write(f"Energy Range: {uv_vis_plot_params.energyRange}\n")
        f.write(f"Points: {uv_vis_plot_params.points}\n")
        f.write(f"Standard Deviation: {uv_vis_plot_params.std_dev}\n")
        f.write(f"Monomer Energies: {frenkel_params.eps}\n")
        f.write(f"Monomer Dipole Moments: {frenkel_params.mu}\n")
        f.write(f"Dipole-Dipole Coupling Coefficient: {frenkel_params.J_DD}\n")
        f.write(f"Superexchange Coupling Coefficient: {frenkel_params.J_SE}\n")
        f.write(f"Distance between Monomers: {R}\n\n")

        H = uv_vis_spectrum_generator._generateHamiltonian(sequence)
        f.write(f"Hamiltonian Matrix:\n{np.round(H, 3)}\n\n")

        eigenvalues, eigenvectors = np.linalg.eig(H)
        f.write(f"Eigenvalues (Exciton Energies):\n{np.round(eigenvalues, 3)}\n\n")
        f.write(f"Eigenvectors (Exciton States):\n{np.round(eigenvectors, 3)}\n\n")
        eigen_fig = plot_eigenvalue(eigenvectors, eigenvalues, sequence)
        eigen_fig.savefig(os.path.join(path, 'eigenvectors.svg'))
        plt.close()

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
    ax.set_xlim(uv_vis_plot_params.energyRange)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid()
    ax.set_axisbelow('on')
    fig.savefig(os.path.join(path, 'uv_vis_spectrum.svg'))
    plt.close()

    nmr_file = 'nmr.txt'
    with open(os.path.join(path, nmr_file), 'w') as f:
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
    fig.savefig(os.path.join(path, 'nmr_spectrum.svg'))
    plt.close()

    ms_file = 'ms.txt'
    with open(os.path.join(path, ms_file), 'w') as f:
        f.write(f"Mass Spectrum for sequence: {sequence}\n")
        f.write(f"Mass Range: {ms_plot_parameters.massRange}\n")
        f.write(f"Bin Width: {ms_plot_parameters.binWidth}\n\n")

        f.write(f"Dropout: {ms_noise_parameters.dropout}\n")
        f.write(f"Weight: {ms_noise_parameters.weight}\n")
        f.write(f"Extra Peaks: {ms_noise_parameters.extra_peaks}\n")
        f.write(f"Noise Width: {ms_noise_parameters.width}\n\n")

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
    ax.bar(x, y, width=ms_plot_parameters.binWidth, color=colors["MS"], label='Mass Spectrum')
    ax.set_xlabel('Mass [m/z]')
    ax.set_ylabel('Intensity')
    ax.set_xlim(ms_plot_parameters.massRange[0] - ms_plot_parameters.binWidth, max(mass_spectrum.keys()) + ms_plot_parameters.binWidth)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid(axis='y')
    ax.set_axisbelow('on')
    fig.savefig(os.path.join(path, 'ms_spectrum.svg'))
    plt.close()

if __name__ == '__main__':
    main()
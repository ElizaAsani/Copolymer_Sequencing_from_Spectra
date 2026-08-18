"""
Optical spectra generator
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class FrenkelParameters:
    """data class for Frenkel parameters"""
    monomers : list
    eps : list                      # excited state energy for each monomer [eV]
    mu : list                       # transition dipole moment for each monomer [D]
    J_DD : float                    # through-space dipole-dipole coupling constant [eV/D^2]
    J_SE : float                    # nearest-neighbor superexchange coupling constant [eV]

@dataclass
class GaussianPlotParameters:
    """data class for gaussian plot parameters"""
    points : int                    # number of points on the curve
    std_dev : float                 # standard deviation of Gaussian peak
    energy_range : list=None         # energy range of UV-Vis spectrophotometer
    wavelength_range : list=None     # wavelength range of UV-Vis spectrophotometer
    peaks : int=None                # number of highest-intensity peaks to retain

#----Dipole-Dipole Coupling Parameters-----#

R = 5 * 10**-10                     # distance between monomers [m] (5 A)
eps_r = 1.0                         # relative permittivity (1.0)

class UV_Vis:
    """Class to generate the UV-Vis spectrum of a copolymer chain using the Frenkel Hamiltonian model.
    """
    # J_DD calculation constants
    eps_0 = 8.854 * 10**-12     # vacuum permittivity [F/m] (8.854 * 10**-12)
    DtoCm = 3.336 * 10**-30     # conversion factor from Debye to Coulomb-meter (3.336 * 10**-30)
    JtoeV = 6.242 * 10**18      # conversion factor from Joules to electron-volts (6.242 * 10**18)

    # converting between eV and nm
    h = 4.136 * 10**-15         # Planck's contant [eV*s]
    c = 2.9979 * 10**17         # speed of light [nm/s]

    # conversion between eV to nm or vice versa
    def ev_to_nm(unit):
        return UV_Vis.h*UV_Vis.c / unit

    def __init__(self, Frenkel_parameters : FrenkelParameters, plot_parameters : GaussianPlotParameters):
        self.monomers = Frenkel_parameters.monomers
        self.eps = Frenkel_parameters.eps       
        self.mu = Frenkel_parameters.mu
        self.J_DD = Frenkel_parameters.J_DD
        self.J_SE = Frenkel_parameters.J_SE
        
        self.points = plot_parameters.points
        self.peaks = plot_parameters.peaks
        self.std_dev = plot_parameters.std_dev                
        self.energy_range = plot_parameters.energy_range
        self.wavelength_range = plot_parameters.wavelength_range
        
        if self.energy_range is not None:
            self.unit = 'energy'
            self.x = np.linspace(self.energy_range[0], self.energy_range[1], self.points)
            self.x_energy = self.x
        elif self.wavelength_range is not None:
            self.unit = 'wavelength'
            self.energy_range = [UV_Vis.ev_to_nm(self.wavelength_range[1]), UV_Vis.ev_to_nm(self.wavelength_range[0])]
            self.x = np.linspace(self.wavelength_range[0], self.wavelength_range[1], self.points)
            self.x_energy = UV_Vis.ev_to_nm(self.x)
        else:
            raise TypeError("Must specify either energy range or wavelength range for the spectrum.")

    @classmethod
    def calculateJDD(cls, R, eps_r):
        return np.round(- 2 * cls.DtoCm**2 / (4 * np.pi * eps_r * cls.eps_0 * R**3) * cls.JtoeV, 2)

    def _generateHamiltonian(self, sequence):
        """Generates a matrix representation of the Hamiltonian for a copolymer 
            chain of given length. 
        """
        length = len(sequence)

        # initialize Hamiltonian
        H = np.zeros((length, length))

        # fill diagonals with energy of each monomer
        for i in range(length):
            monomer = self.monomers.index(sequence[i])
            H[i][i] = self.eps[monomer]

        # fill off-diagonals with intersite exciton coupling
        for i in range(length - 1):
            for j in range(i + 1, length):
                monomer_1 = self.monomers.index(sequence[i])
                monomer_2 = self.monomers.index(sequence[j])
                dist = j - i
        
                H[i][j] = self._calcIntersiteCoupling(monomer_1, monomer_2, dist)
                H[j][i] = self._calcIntersiteCoupling(monomer_2, monomer_1, dist)

        return H

    def _calcIntersiteCoupling(self, monomer_1, monomer_2, dist):
        """Calculates the intersite exciton coupling between two monomers.
        """

        # calculate dipole-dipole coupling
        dip_dip = self.J_DD * self.mu[monomer_1]*self.mu[monomer_2] / ((dist)**3)

        # add superexchange coupling if nearest neighbor
        if (dist == 1):
            dip_dip += self.J_SE
            
        return dip_dip

    def _generateAbsorption(self, eigenvalues, eigenvectors, sequence):
        """Generates an array of absorption values with their corresponding 
            intensities given the eigenvalues and eigenvectors of a sequence. 
        """
        
        length = len(sequence)

        # generate array of dipole moments for each monomer
        dipole_moments = np.zeros(length)
        for i in range(length):
            monomer = self.monomers.index(sequence[i])
            dipole_moments[i] = self.mu[monomer]

        # generate dictionary to store absorption values and intensities
        absorption = {}

        # calculate the intensities of each corresponding absorption energy
        for i in range(length):
            dipole_product = dipole_moments * eigenvectors[:,i]
            intensity = (np.sum(dipole_product))**2
            absorption[eigenvalues[i]] = intensity

        # sort by intensity and select top peaks, if needed
        if self.peaks is not None:
            absorption = dict(sorted(absorption.items(), key=lambda item:item[1], reverse=True)[:self.peaks])        

        # sort absorption from lowest to highest energy
        absorption = pd.DataFrame((sorted(absorption.items())), columns=['energy', 'intensity']) 

        return absorption

    def _generateGaussian(self, absorption, normalize=False):
        """Generates absorption values for a gaussian representation of the 
            absorption spectrum withing an energy range.
        """
        # compute each y-coordinate by sum of gaussian curves for each absorption
        y = np.zeros(self.points)

        for energy, intensity in absorption.values:
            y += intensity * np.exp(-(self.x_energy - energy)**2/(2*self.std_dev**2))

        # normalize the spectrum if requested
        if normalize:
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

        return y

    def getAbsorption(self, sequence):
        """Wrapper function to get absorption energies and intensities"""
        H = self._generateHamiltonian(sequence)
        eigenvalues, eigenvectors = np.linalg.eig(H)
        absorption = self._generateAbsorption(eigenvalues, eigenvectors, sequence)

        # cutoff absorption outside of range
        absorption = absorption[(absorption['energy'] >= self.energy_range[0]) & (absorption['energy'] <= self.energy_range[1])]

        # add wavelength 
        absorption['wavelength'] = UV_Vis.ev_to_nm(absorption['energy'])

        return absorption

    def getSpectrum(self, sequence, normalize=False):
        """Wrapper function to generate the absorption spectrum for a given sequence.
        """
        # generate the Hamiltonian for the sequence and solve
        H = self._generateHamiltonian(sequence)
        eigenvalues, eigenvectors = np.linalg.eig(H)

        # generate the absorption for the sequence
        absorption = self._generateAbsorption(eigenvalues, eigenvectors, sequence)

        # generate points on absorption curve from gaussian
        y = self._generateGaussian(absorption, normalize)

        return y
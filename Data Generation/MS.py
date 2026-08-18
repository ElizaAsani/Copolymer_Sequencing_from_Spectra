"""
Mass spectrum generator
"""

import numpy as np

from dataclasses import dataclass
from enum import Enum

@dataclass
class MSParameters:
    """data class for mass spectrum parameters"""
    monomers : list         # list of monomers + endgroup ('*')
    formulas : list         # chemical formulas of all monomers and endgroup
    s : float = 6.0         # width of Gaussian distribution of intensities over breaking points

@dataclass
class BarPlotParameters:
    """data class for bar plot parameters"""
    mass_range : list        # range of mass spectrometer [m/z]
    bin_width : int          # width of bins used to histogram spectrum [m/z] 

@dataclass
class MSNoiseParameters:
  """data class for mass spectrum noise parameters"""
  dropout : float = 0.0     # dropout probability for each peak
  extra_peaks : int = 0     # number of satellite peaks to generate around each peak
  width : float = 182.0     # width of linear intensity decay for satellite peak relative to major peak [m/z]
  weight : float = 1.0      # lower bound for reweighting each peak

@dataclass
class Atom:
    """data class for elements"""
    name : str
    mass : float

class Element(Enum):
    """Enum for elements with their isotopic distributions."""
    H = Atom(name='H', mass = 1)
    C = Atom(name='C', mass = 12)
    N = Atom(name='N', mass = 14)
    O = Atom(name='O', mass = 16)
    S = Atom(name='S', mass = 32)

def gaussianPDF(x, mu, sigma):
    coef = 1 / (sigma * np.sqrt(2*np.pi))
    z = (x - mu) / sigma
    return coef * np.exp(-(z**2)/2)

class MS:
    """Class to generate the mass spectrum of a copolymer chain.
    """

    def __init__(self, MS_parameters : MSParameters, bar_plot_params : BarPlotParameters, noise_params : MSNoiseParameters):
        self.rngs = {'dropout': np.random.default_rng(12),
                    'extra_peaks': np.random.default_rng(24), 
                    'reweighting': np.random.default_rng(48)}

        self.monomers = {}
        for mon, form in zip(MS_parameters.monomers, MS_parameters.formulas):
          self.monomers[mon] = self._getFormulaMass(form)

        self.end_char = MS_parameters.monomers[-1]  # assuming last monomer is end character

        self.s = MS_parameters.s

        # create dictionary of atoms that can be removed from each peak
        monomer_formulas = MS_parameters.formulas[:-1]
        self.common_formula = monomer_formulas[0]
        for i in range(1, len(monomer_formulas)):
          monomer_formula = monomer_formulas[i]
          self.common_formula = {k: min(self.common_formula[k], monomer_formula[k]) for k in self.common_formula.keys() & monomer_formula.keys()}

        self.dropout = noise_params.dropout
        self.extra_peaks = noise_params.extra_peaks
        self.noise_width = noise_params.width
        self.peak_weight = noise_params.weight

        self.mass_range = bar_plot_params.mass_range
        self.bin_width = bar_plot_params.bin_width
        self.x = np.arange(self.mass_range[0], self.mass_range[1], self.bin_width)

    def getSpectrum(self, seq):
        """Generates the binned mass spectrum for a given copolymer sequence.
        """
        mass_spectrum = self.generateMassSpectrum(seq)

        # bin the mass spectrum
        intensities = np.zeros_like(self.x, dtype=float)
        for m, inten in mass_spectrum.items():
            if self.mass_range[0] <= m < self.mass_range[1]:
                bin_index = int((m - self.mass_range[0]) / self.bin_width)
                intensities[bin_index] += inten
            else:
                print(f"Mass {m} out of range {self.mass_range}, skipping.")

        # normalize to max intensity
        intensities = intensities / max(intensities)

        return intensities

    def generateMassSpectrum(self, sequence):
        """Generates the mass spectrum for a given copolymer sequence.
        """
        fragments = self._getFragments(sequence)
        mass_spectrum = {}
        for fragment, count in fragments.items():
            mass = self._getFragmentMass(fragment)
            mass_spectrum[mass] = mass_spectrum.get(mass, 0) + count

        # add noise
        mass_spectrum = self._addNoise(mass_spectrum)

        # sort by mass
        mass_spectrum = dict(sorted(mass_spectrum.items()))

        return mass_spectrum
    
    def _getFragments(self, sequence):
        """Generates Gaussian-distributed fragments centered at midpoint of sequence."""

        sequence = self.end_char + sequence + self.end_char
        length = len(sequence)
        mid_point = length / 2
        fragments = {sequence : float(0)}
        for i in range(length):
            prob = gaussianPDF(i, mid_point, self.s)
            if i <= 1 or i >= length - 1:
                # case of no break, or unstable endgroup break
                fragments[sequence] = fragments.get(sequence, 0) + prob
            else:
                fragments[sequence[:i]] = prob
                fragments[sequence[i:]] = prob

        return fragments

    def _getFragmentMass(self, fragment):
        """Calculate the mass of a fragment"""

        mass = 0
        monomer_counts = {mon : fragment.count(mon) for mon in self.monomers.keys()}
        for monomer, count in monomer_counts.items():
            mass += self.monomers[monomer] * count

        return mass

    def _getFormulaMass(self, formula):
        """Calculate the mass of a chemical formula"""

        mass = 0
        for element, count in formula.items():
            elem = Element[element]
            mass += elem.value.mass * count

        return mass

    def _addNoise(self, mass_spectrum):
        """Add noise to the spectrum through extra fragmentation, dropout, and peak reweighting """
        
        if self.extra_peaks > 0:
          # add extra fragmentation
          extra_fragments = {}
          for mass, intensity in mass_spectrum.items():
              # generate random subsets of atoms to subtract (without losing an entire monomer)
              extra_formulas = {k : self.rngs['extra_peaks'].integers(1, self.common_formula[k], size=self.extra_peaks) for k in self.common_formula}
              extra_form_masses = {k: Element[k].value.mass*extra_formulas[k] for k in extra_formulas}
              extra_masses = mass - np.add.reduce(list(extra_form_masses.values()))
              # ensure new masses are in range
              extra_masses = np.array([extra_mass for extra_mass in extra_masses if (extra_mass >= self.mass_range[0] and extra_mass <= self.mass_range[1])])
              # get new intensity as fraction of major peak intensity
              extra_intensities = ((mass - extra_masses)/self.noise_width)*intensity
              extra_fragments.update(dict(zip(extra_masses, extra_intensities)))

          mass_spectrum.update(extra_fragments)

        if self.dropout > 0:
            # randomly drop peaks
            ms_keys = list(mass_spectrum.keys())[:-1]
            for mass in ms_keys:
                drop = self.rngs['dropout'].uniform(0, 1) < self.dropout
                if drop: del mass_spectrum[mass]

        if (self.peak_weight >= 0 and self.peak_weight < 1):
          # randomly alter peak intensities
          for mass, _ in mass_spectrum.items():
            mass_spectrum[mass] *=  self.rngs['reweighting'].uniform(self.peak_weight, 1)

        return mass_spectrum

    def getMolecularIon(self, sequence):
        """Calculate the molecular ion mass of the full sequence."""
        sequence = self.end_char + sequence + self.end_char
        return self._getFragmentMass(sequence)
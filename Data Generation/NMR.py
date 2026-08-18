"""
NMR spectra generator
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class NMRPlotParameters:
    """data class for lorentzian plot parameters"""
    points : int                # number of points on the curve
    half_width : float          # half-width of the Lorentzian peak
    shift_range : list          # absolute chemical shift range [ppm]  
    reference_shift : float     # reference chemical shift [ppm]
    tolerance : float = 0.0     # tolerance for peak consolidation 

class NMR:
    """Class to generate the NMR spectrum of a copolymer chain.
    """

    def __init__(self, trimers_file, dimers_file, plot_parameters : NMRPlotParameters):       
        
        self.points = plot_parameters.points
        self.half_width = plot_parameters.half_width
        self.shift_range = plot_parameters.shift_range
        self.reference_shift = plot_parameters.reference_shift 
        self.tolerance = plot_parameters.tolerance
        self._preprocessShiftRange()

        self.x = np.linspace(self.shift_range[0], self.shift_range[1],  self.points)

        self.nmr_trimers = self._preprocessNMRSubgroups(NMR.readNMRSubgroups(trimers_file))
        self.nmr_endgroups = self._preprocessNMRSubgroups(NMR.readNMRSubgroups(dimers_file))

    @staticmethod
    def readNMRSubgroups(filename):
        """Reads NMR subgroups from a CSV file."""
        nmr_subgroups = pd.read_csv(filename, index_col=0)

        # convert string lists of shifts and intensities to lists of floats
        nmr_subgroups['shifts'] = nmr_subgroups['shifts'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
        nmr_subgroups['intensities'] = nmr_subgroups['intensities'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])

        return nmr_subgroups
    
    def _consolidateShifts(self, shifts, intensities):
        """Consolidates shifts and intensities by merging close shifts."""
        consolidated_shifts = []
        consolidated_intensities = []

        # sort shifts and intensities by shift
        sorted_indices = np.argsort(shifts)
        shifts = np.array(shifts)[sorted_indices]
        intensities = np.array(intensities)[sorted_indices]

        cur_shifts = []
        cur_intensity = 0
        # iterate over shifts and intensities
        for shift, intensity in zip(shifts, intensities):
            # check if shift is close to any existing shift in the current list
            if (abs(cur_shifts - shift) < self.tolerance).any():
                # if close, add shift to list and add intensity
                cur_shifts.append(shift)
                cur_intensity += intensity
            else:
                # if not close, consolidate current shifts and intensity
                if cur_shifts:
                    consolidated_shifts.append(np.mean(cur_shifts))
                    consolidated_intensities.append(cur_intensity)
                # reset current shifts and intensity
                cur_shifts = [shift]
                cur_intensity = intensity

        # add the last consolidated shift and intensity
        consolidated_shifts.append(np.mean(cur_shifts))
        consolidated_intensities.append(cur_intensity)

        return consolidated_shifts, consolidated_intensities
    
    def _getRelativeShift(self, shift):
        """Calculates the relative shift in ppm."""
        return np.round(self.reference_shift - shift, 3)
    
    def _preprocessNMRSubgroups(self, nmr_subgroups):
        """Preprocesses NMR subgroups by consolidating shifts and converting to ppm."""

        for subgroup in nmr_subgroups.index:
            shifts, intensities = nmr_subgroups.loc[subgroup]['shifts'], nmr_subgroups.loc[subgroup]['intensities']

            # consolidate shifts and intensities
            shifts, intensities = self._consolidateShifts(shifts, intensities)

            # calculate relative shifts
            shifts = [self._getRelativeShift(shift) for shift in shifts]

            # update the NMR subgroups DataFrame
            nmr_subgroups.at[subgroup, 'shifts'] = shifts
            nmr_subgroups.at[subgroup, 'intensities'] = intensities
        
        return nmr_subgroups
    
    def _preprocessShiftRange(self):
        """Shift the plot range."""
        self.shift_range = [self._getRelativeShift(shift) for shift in self.shift_range]

        return
    
    def _countTrimers(self, seq):
        """Counts the number of trimers in a sequence."""
        trimers = {key: 0 for key in self.nmr_trimers.index}

        # count trimers in sequence
        for i in range(len(seq) - 2):
            trimer = seq[i:i+3]
            # check if trimer is in dictionary
            if trimer in self.nmr_trimers.index:
                trimers[trimer] += 1
            # check if reversed trimer is in dictionary
            elif trimer[::-1] in self.nmr_trimers.index:
                trimers[trimer[::-1]] += 1

        return trimers

    def _countEndgroups(self, seq):
        """Counts which endgroups are in a sequence."""
        endgroups = {key: 0 for key in self.nmr_endgroups.index}

        # count endgroups in sequence
        n = 2
        start = seq[0:n]  
        end = seq[-n:][::-1] # reverse end to match endgroup orientation

        endgroups[start] += 1
        endgroups[end] += 1

        return endgroups

    def _generateSpectrum(self, trimers, endgroups):
        """Generates the NMR spectrum for a given sequence."""
        # initialize dictionary to store shifts and intensities
        spectrum = {}
    
        # helper to accumulate shifts/intensities from a DataFrame
        def accumulate(df, counts):
            for subgroup in df.index:
                if counts[subgroup] == 0:
                    continue
                row = df.loc[subgroup]
                for shift, intensity in zip(row['shifts'], row['intensities']):
                    if shift in spectrum.keys():
                        # if shift already exists, add intensity to existing value
                        spectrum[shift] += counts[subgroup] * intensity
                    else:
                        # if shift does not exist, create new entry
                        spectrum[shift] = counts[subgroup] * intensity

        # Accumulate for trimers and endgroups
        accumulate(self.nmr_trimers, trimers)
        accumulate(self.nmr_endgroups, endgroups)

        # sort spectrum from lowest to highest shift
        spectrum = pd.DataFrame(sorted(spectrum.items()), columns=['shift', 'intensity'])

        return spectrum
    
    def _generateLorentzian(self, spectrum, normalize):
        """Generates a Lorentzian-broadened NMR spectrum for a given sequence."""
        # compute each y-coordinate by sum of Lorentzian curves for each shift
        y = np.zeros(self.points)

        # normalize the intensities (peak areas) if requested
        if normalize:
            spectrum['intensity'] = spectrum['intensity'] / np.sum(spectrum['intensity'])

        # Generate the spectrum using Lorentzian broadening
        for shift, intensity in spectrum.values:
            y += (intensity/np.pi) * self.half_width / ((self.x - shift)**2 + self.half_width**2)

        return y      

    def getSpectrum(self, seq, normalize=True):
        """Wrapper function to generate the NMR spectrum for a given sequence."""
        
        # get counts of each trimer in the sequence
        trimers = self._countTrimers(seq)

        # add endgroups to the counts
        endgroups = self._countEndgroups(seq)
        
        # generate the aggregate shifts and intensities for the spectrum
        spectrum = self._generateSpectrum(trimers, endgroups)

        # generate points on NMR curve from Lorentzian-broadened spectrum
        y = self._generateLorentzian(spectrum, normalize)

        return y
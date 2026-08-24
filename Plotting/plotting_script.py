import os

import matplotlib.pyplot as plt

from plot_reconstruction import plot_sequence_reconstruction
from plot_degeneracies import plot_degeneracies

plt.style.use('figure.mplstyle')

palette = ['#6A4E5A','#A72420', '#909033', '#DDC0A5', '#E0884D']

colors = {
    "NMR": palette[0],          
    "MS": palette[1],            
    "UV-Vis": palette[2],    

    "UV-Vis + MS": palette[3],  
    "UV-Vis + MS + NMR": palette[4]                
}

def build_model_folders(models, models_root, noise_level):
    return {spectrum: os.path.join(models_root, folder, noise_level) 
            for spectrum, folder in models.items()}

models_root = os.path.join("../Multispectra-to-sequence Transformer", "Output")
degeneracy_root = os.path.join("../Data Generation", "degeneracy")

models = {"NMR": "nmr",
          "MS": "ms",
          "UV-Vis": "uv_vis",
          "UV-Vis + MS": "multispectra/uv_vis+ms",
          "UV-Vis + MS + NMR": "multispectra/all/"}

noise_levels = ["NOISE0", "NOISE1", "NOISE2"]

alpha = 1

folders_by_noise = {noise_level: build_model_folders(models, models_root, noise_level) for noise_level in noise_levels}
degeneracy_folders = {spectrum: os.path.join(degeneracy_root, models[spectrum]) for spectrum in ["NMR", "MS", "UV-Vis"]}

## ---- Figure 3 ---- ##
plot_degeneracies(degeneracy_folders, colors)

## ---- Figure 4 ---- ##
plot_sequence_reconstruction(folders_by_noise[noise_levels[0]], colors, alpha=alpha, degeneracy_folders=degeneracy_folders)
plot_sequence_reconstruction(folders_by_noise[noise_levels[1]], colors, alpha=alpha)
plot_sequence_reconstruction(folders_by_noise[noise_levels[2]], colors, alpha=alpha)

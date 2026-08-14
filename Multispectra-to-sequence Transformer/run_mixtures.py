"""
Code to run sequence generation from spectra
"""

import os
import torch
from torch.utils.data import DataLoader

from SequenceEncoder import SeqDatasetMixtures
from model import build_multi_spectra_transformer
from decode import beam_search_mixtures

# initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# model path
path = rf"./Output/multispectra/all/NOISE0/"
saved_model_path = path + r"model_dict"  
scale = {'uv_vis': True, 'nmr': False, 'ms': False}
spec_lengths = {'uv_vis': 220, 'nmr': 500, 'ms': 356}

# initialize model
transformer = build_multi_spectra_transformer(spec_lengths=spec_lengths, seq_length=21, vocab_size=5, 
                                              d_model=128, h=4, N=3, d_ff=256, dropout=0.1)
transformer.to(device)

# load saved model
transformer.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location=device))
transformer.eval()

# beam search parameters
beam_width = 10
alpha = 1
threshold = 1

# directory to store mixtures output
mix_path = path + f"mixtures_L10/alpha_{alpha}/"
if not os.path.exists(mix_path):
    os.makedirs(mix_path)

lambdas = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]

for lamb in lambdas:
    out_path = mix_path + f"lamb_{lamb}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    data_file = f"./mixtures/mixtures_L10_lamb{lamb}_NOISE0.h5"

    mixtures_dataset = SeqDatasetMixtures(data_file, scale)

    beam_df = beam_search_mixtures(transformer, mixtures_dataset, device=device, beam_width=beam_width, alpha=alpha)
    beam_df.to_csv(out_path + "beam_search.csv", index=False)
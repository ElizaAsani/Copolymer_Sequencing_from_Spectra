"""
Code to run sequence generation from spectra
"""

import os
import argparse

import torch

from SequenceEncoder import SeqDatasetMixtures
from model import build_multi_spectra_transformer
from beam_search import beam_search_mixtures

from config import load_mix_config, load_model_architecture

def parse_args():
    parser = argparse.ArgumentParser(description="Run beam search inference on mixtures data using saved model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()

args = parse_args()
cfg = load_mix_config(args.config)

# initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# model path
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.normpath(os.path.join(script_dir, cfg.output_dir))
saved_model_path = os.path.join(out_path, "model_dict")
model_cfg = load_model_architecture(os.path.join(out_path, "model_config.yaml"))

scale = cfg.scale
max_model_length = model_cfg.seq_length + 1

# initialize model
transformer = build_multi_spectra_transformer(spec_lengths=model_cfg.spec_lengths, seq_length=max_model_length, 
                                              vocab_size=model_cfg.vocab_size, d_model=model_cfg.d_model, h=model_cfg.h, 
                                              N=model_cfg.N, d_ff=model_cfg.d_ff, dropout=model_cfg.dropout)
transformer.to(device)

# load saved model
transformer.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location=device))
transformer.eval()

# beam search parameters
beam_width = cfg.beam_width
alpha = cfg.alpha
threshold = 1

# directory to store mixtures output
mix_path = os.path.join(out_path, f"mixtures_L10/alpha_{alpha}/")
if not os.path.exists(mix_path):
    os.makedirs(mix_path)

lambdas = cfg.mixtures_lambdas

for lamb in lambdas:
    out_path = os.path.join(mix_path, f"lambda_{lamb}/")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    data_file = cfg.mixtures_file_template.format(lamb=lamb)

    mixtures_dataset = SeqDatasetMixtures(data_file, model_cfg.seq_length, scale)

    beam_df = beam_search_mixtures(transformer, mixtures_dataset, device=device, beam_width=beam_width, alpha=alpha)
    beam_df.to_csv(os.path.join(out_path, "beam_search.csv"), index=False)
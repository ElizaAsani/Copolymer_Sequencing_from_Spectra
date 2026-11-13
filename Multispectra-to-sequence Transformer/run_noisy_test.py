"""
Code to run sequence generation from spectra
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
import pandas as pd

from SequenceEncoder import SeqDataset, make_splits
from model import build_multi_spectra_transformer
from evaluate import get_accuracy
from decode import inference
from beam_search import beam_search

from config import load_noisy_test_config, load_model_architecture

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on noisy test data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()

args = parse_args()
cfg = load_noisy_test_config(args.config)

# initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# model path
script_dir = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.normpath(os.path.join(script_dir, cfg.model_dir, "model_dict"))
model_cfg = load_model_architecture(os.path.join(script_dir, cfg.model_dir, "model_config.yaml"))
max_model_length = model_cfg.seq_length + 1

# initialize model
transformer = build_multi_spectra_transformer(spec_lengths=model_cfg.spec_lengths, seq_length=max_model_length, 
                                              vocab_size=model_cfg.vocab_size, d_model=model_cfg.d_model, h=model_cfg.h, 
                                              N=model_cfg.N, d_ff=model_cfg.d_ff, dropout=model_cfg.dropout)
transformer.to(device)

# load saved model
transformer.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location=device))
transformer.eval()

# directory to store output
out_path = os.path.normpath(os.path.join(script_dir, cfg.model_dir, cfg.output_dir))
if not os.path.exists(out_path):
    os.makedirs(out_path)

# generate test data
data_file = os.path.normpath(os.path.join(script_dir, cfg.data_file))
scale = cfg.scale

dataset = SeqDataset(data_file, scale, max_model_length)
_, _, test_set = make_splits(dataset)
batch_size = 64 # hard-coded for now, can be changed later
test_dl = DataLoader(test_set, batch_size=4*batch_size)
print(f"Number of test sequences: {len(test_set)}")
print(f"Number of test batches: {len(test_dl)}")

target_sequences, predicted_sequences, sequence_lengths, lambdas = inference(transformer, test_dl, device=device)

output_path = os.path.join(out_path, "output.txt")
with(open(output_path, "w")) as f:

    seq_acc, char_acc, char_per_seq_acc, wrong_lengths, errors = get_accuracy(predicted_sequences, target_sequences)
    f.write(f"Percent sequences reconstructed exactly: {seq_acc}\n")
    f.write(f"Percent characters reconstructed exactly: {char_acc}\n")
    f.write(f"Percent characters reconstructed per sequence: {char_per_seq_acc}\n")
    f.write(f"Number of sequences with different lengths: {wrong_lengths}\n")

# error analysis
errors_path = os.path.join(out_path, "errors")
if not os.path.exists(errors_path):
    os.makedirs(errors_path)
errors_file_path = os.path.join(errors_path, "errors.csv")
errors_df = pd.DataFrame({
    "target_sequence": target_sequences,
    "predicted_sequence": predicted_sequences,
    "sequence_length": sequence_lengths,
    "num_errors": errors
})
errors_df.to_csv(errors_file_path, index=False)

# beam search
beam_width = cfg.beam_width
alpha = cfg.alpha

beam_search_path = os.path.join(out_path, f"beam_search/alpha_{alpha}")
if not os.path.exists(beam_search_path):
    os.makedirs(beam_search_path)
beam_search_file_path = os.path.join(beam_search_path, "beam_search.txt")
beam_search_csv_path = os.path.join(beam_search_path, "beam_search.csv")

test_dl = DataLoader(test_set, batch_size=max(1, 4*(batch_size//beam_width)))
beam_df, percent_hits, percent_top_5_hits, percent_top_1_hits = beam_search(transformer, test_dl, device=device, beam_width=beam_width, alpha=alpha)

f = open(beam_search_file_path, "w")
print(f"Number of sequences: {len(test_set)}", file=f)
print(f"Beam width: {beam_width}", file=f)
print(f"Alpha: {alpha}", file=f)
print(f"Percent top {beam_width} hits: {percent_hits}%", file=f)
print(f"Percent top 5 hits: {percent_top_5_hits}%", file=f)
print(f"Percent top 1 hits: {percent_top_1_hits}%", file=f)
f.close()

beam_df.to_csv(beam_search_csv_path, index=False)

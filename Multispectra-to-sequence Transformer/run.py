"""
Code to run sequence generation from spectra
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from SequenceEncoder import SeqDataset, make_splits
from model import build_multi_spectra_transformer
from train import train_model
from evaluate import get_accuracy
from decode import inference
from beam_search import beam_search, plot_beam_histogram

from plots import plot_errors

from config import load_config, write_model_architecture

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate multispectra-to-sequence transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()

args = parse_args()
cfg = load_config(args.config)

# initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# directory to store output
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.normpath(os.path.join(script_dir, cfg.data_file))
out_path = os.path.normpath(os.path.join(script_dir, cfg.output_dir))
if not os.path.exists(out_path):
    os.makedirs(out_path)
chkpt_path = os.path.join(out_path, "chkpts")
if not os.path.exists(chkpt_path):
    os.makedirs(chkpt_path)
saved_model = cfg.saved_model
saved_model_path = os.path.join(out_path, "model_dict")
output_path = os.path.join(out_path, "output.txt")
model_path = os.path.join(out_path, "model.txt")

# load dataset
scale = cfg.scale
dataset = SeqDataset(data_file, scale)

# initialize model
transformer = build_multi_spectra_transformer(spec_lengths=dataset.spectra_length,
                                seq_length=dataset.model_max_sequence_length,
                                vocab_size=dataset.num_chars, 
                                d_model=cfg.d_model, h=cfg.h, N=cfg.N, d_ff=cfg.d_ff, dropout=cfg.dropout)
transformer.to(device)

# write model architecture to config file
model_config_path = os.path.join(out_path, "model_config.yaml")
model_config = {
    "spec_lengths": {key: int(value) for key, value in dataset.spectra_length.items()},
    "seq_length": int(dataset.max_sequence_length),
    "vocab_size": int(dataset.num_chars),
    "d_model": cfg.d_model,
    "h": cfg.h,
    "N": cfg.N,
    "d_ff": cfg.d_ff,
    "dropout": cfg.dropout
}
write_model_architecture(model_config_path, model_config)

# training params
epochs = cfg.epochs   
batch_size = cfg.batch_size

# generate train/test data
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = make_splits(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
train_dl = DataLoader(train_set, batch_size=batch_size)
val_dl = DataLoader(val_set, batch_size=4*batch_size)
test_dl = DataLoader(test_set, batch_size=4*batch_size)

# use saved model or train new one
if (saved_model):
    transformer.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location=device))
    transformer.eval()
else:
    f = open(model_path, "w")

    # write model info to file
    print(f"Data file: {data_file}", file=f)
    print(f"Number of samples: {len(dataset)}", file=f) 
    print(f"Train/Val/Test split: {len(train_set)}/{len(val_set)}/{len(test_set)}", file=f)
    print(f"Scaling: {dataset.scale}", file=f)
    print(transformer, file=f)
    f.close()

    train_model(transformer, train_dl, val_dl, epochs, cfg.d_model, chkpt_path, device=device).savefig(os.path.join(out_path, "loss.png"))
    torch.save(transformer.state_dict(), saved_model_path)
    transformer.eval()

target_sequences, predicted_sequences, sequence_lengths, lambdas = inference(transformer, test_dl, device=device)

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
errors_df, errors_fig = plot_errors(target_sequences, predicted_sequences, sequence_lengths, errors)
errors_df.to_csv(errors_file_path, index=False)
errors_fig.savefig(os.path.join(errors_path, "errors.png"))

# beam search
beam_width = cfg.beam_width
alpha = cfg.alpha
threshold = 1

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
plot_beam_histogram(beam_search_csv_path, threshold=threshold).savefig(os.path.join(beam_search_path, f"beam_histogram_thresh_{threshold}.png"))
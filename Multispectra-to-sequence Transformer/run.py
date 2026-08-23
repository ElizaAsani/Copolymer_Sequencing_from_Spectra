"""
Code to run sequence generation from spectra
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import DataLoader

from SequenceEncoder import SeqDataset, make_splits
from model import build_multi_spectra_transformer
from train import train_model, get_accuracy
from decode import inference, beam_search, plot_beam_histogram

from plots import plot_errors

# initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# directory to store output
data_file = "multispectra_NOISE2.h5"
path = r"./Output/multispectra/all/NOISE2"
if not os.path.exists(path):
    os.makedirs(path)
chkpt_path = path + r"/chkpts"
if not os.path.exists(chkpt_path):
    os.makedirs(chkpt_path)
saved_model = False
saved_model_path = path + r"/model_dict"   
output_path = path + r"/output.txt"
model_path = path + r"/model.txt"

# load dataset
scale = {'uv_vis': True, 'nmr': False, 'ms': False}
dataset = SeqDataset(data_file, scale)

# transformer parameters 
d_model = 128
h=4
N=3
d_ff = 256
dropout = 0.1

# initialize model
transformer = build_multi_spectra_transformer(spec_lengths=dataset.spectra_length,
                                seq_length=dataset.model_max_sequence_length,
                                vocab_size=dataset.num_chars, 
                                d_model=d_model, h=h, N=N, d_ff=d_ff, dropout=dropout)
transformer.to(device)

# training params
epochs = 1000   
batch_size = 64 

# generate train/test data
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = make_splits(dataset.sequence_lengths, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
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
    print(f"Spectrum size: {dataset.spectra_length}", file=f)
    print(f"Batch size: {batch_size}", file=f)
    print(f"Model Dimension: {d_model}, ", file=f)
    print(f"Heads: {h}", file=f)
    print(f"Number of Stacked Layers: {N}", file=f)
    print(f"Feed-Forward Layer Dimensions: {d_ff}", file=f)
    print(f"Dropout: {dropout}", file=f)
    print(transformer, file=f)
    f.close()

    train_model(transformer, train_dl, val_dl, epochs, d_model, chkpt_path, device=device).savefig(path + r"/loss.png")
    torch.save(transformer.state_dict(), saved_model_path)
    transformer.eval()

target_sequences, predicted_sequences, sequence_lengths, lambdas = inference(transformer, test_dl, device=device)

with(open(output_path, "w")) as f:

    seq_acc, char_acc, char_per_seq_acc, wrong_lengths, errors = get_accuracy(predicted_sequences, target_sequences)
    f.write(f"Percent sequences reconstructed exactly: {seq_acc}\n")
    f.write(f"Percent characters reconstructed exactly: {char_acc}\n")
    f.write(f"Percent characters reconstructed per sequence: {char_per_seq_acc}\n")
    f.write(f"Number of sequences with different lengths: {wrong_lengths}\n\n")

    for j in range(10):
        f.write(target_sequences[j] + "\n")
        f.write(predicted_sequences[j] + "\n\n")

# error analysis
errors_path = path + r"/errors"
if not os.path.exists(errors_path):
    os.makedirs(errors_path)
errors_file_path = errors_path + r"/errors.csv"
errors_df, errors_fig = plot_errors(target_sequences, predicted_sequences, sequence_lengths, errors)
errors_df.to_csv(errors_file_path, index=False)
errors_fig.savefig(errors_path + r"/errors.png")

# beam search
beam_width = 10
alpha = 1
threshold = 1

beam_search_path = path + rf"/beam_search/alpha_{alpha}"
if not os.path.exists(beam_search_path):
    os.makedirs(beam_search_path)
beam_search_file_path = beam_search_path + r"/beam_search.txt"
beam_search_csv_path = beam_search_path + r"/beam_search.csv"

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
plot_beam_histogram(beam_search_csv_path, threshold=threshold).savefig(beam_search_path + rf"/beam_histogram_thresh_{threshold}.png")
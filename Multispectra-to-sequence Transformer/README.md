# Multispectra-to-sequence Transformer

Trains and evaluates the multispectra-to-sequence transformer.

## Folder structure

```text
.
├── configs/ 
│   ├── run_mixtures.yaml
│   ├── run_NOISE0.yaml
│   ├── run_NOISE1.yaml     
│   └── run_NOISE2.yaml
├── config.py                   # loads configs for run, run_mixtures; writes/loads model_config
├── run.py                      # trains and evaluates multispectra-to-sequence transformer 
├── run_mixtures.py             # beam search inference on mixtures
├── model.py                    # model architecture
├── train.py                    # training script
├── evaluate.py                 # accuracy metrics
├── decode.py                   # greedy decoding 
├── beam_search.py              # beam search
└── SequenceEncoder.py          # dataset class
```

## Usage

```bash
python run.py --config configs/run_NOISE0.yaml
python run_mixtures.py --config configs/run_mixtures.yaml
```

`run.py` trains (or loads a saved model, if `saved_model: true`), evaluates on a held-out test set with greedy decoding, and runs beam search — writing accuracy metrics, errors, and beam search results to `output_dir`.

`run_mixtures.py` loads a model already trained via `run.py` (using its `output_dir`) and runs beam search inference on spectral mixtures data, one output file per sequence correlation (`lambda`) value.

## Data format

Input is an HDF5 file (as produced by the Data Generation pipeline) containing:

- `uv_vis` / `nmr` / `ms` — spectra arrays, one row per sample, for whichever simulators are enabled under `scale` in the config
- `sequence` — copolymer sequence strings, with a `monomers` attribute indicating the vocabulary
- `lambda` — sequence correlation values

Mixtures data is an HDF5 file with a list of sequences and ratios per mixture (see the Data Generation `mixtures` mode output).

## Config

```yaml
io:
  data_file: ../Data Generation/multispectra_NOISE0.h5
  output_dir: ./Output/multispectra/all/NOISE0
  saved_model: false        # if true, load saved model from output_dir instead of training
 
scale:                      # which spectra to use, and whether to min-max scale each
  uv_vis: true
  nmr: false
  ms: false
 
model:
  d_model: 128
  h: 4
  N: 3
  d_ff: 256
  dropout: 0.1
 
training:
  epochs: 1000
  batch_size: 64
 
beam_search:
  beam_width: 10
  alpha: 1                  # length-normalization exponent
```

`configs/run_mixtures.yaml` includes the same `scale`/`beam_search` sections, plus:

```yaml
io:
  mixtures_file_template: ../Data Generation/mixtures_L10/mixtures_lamb{lamb}.h5
  mixtures_lambdas: [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
  output_dir: ./Output/multispectra/all/NOISE0   # must be an output_dir from a completed run.py run
```

The model parameters for mixtures mode are read from `output_dir/model_config.yaml`, which is written for each trained model.

## Evaluation

`decode.py` runs greedy inference, while `beam_search.py` runs a batched beam search, reporting top-1, top-5, and top-*k* hit rates.

`evaluate.py` computes sequence and character reconstruction accuracy for the greedy preditions, which are checked against both the forward and reverse orientation of each target sequence. `evaluate.py` also returns a list of the number of errors in each prediction, which are stored in `output_dir/errors/errors.csv`.

For mixtures, only the beam search results are saved.

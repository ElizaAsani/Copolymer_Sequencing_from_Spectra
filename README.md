# Copolymer_Sequencing_from_Spectra

Code and supporting materials for the manuscript:

**Testing the Limits of Transformer-Based Sequence Inference from Copolymer Spectra**

This repository contains three components:

```text
.
├── Data Generation/                        # simulates DA copolymer sequences and UV-Vis/NMR/MS spectra
├── Multispectra-to-sequence Transformer/   # trains and evaluates the sequence-prediction model
├── Plotting/                               # generates diagnostic and paper figures from the above outputs
└── environment.yaml
```

Each subfolder has its own README with full usage details:

- [`Data Generation/README.md`](./Data%20Generation/README.md)
- [`Multispectra-to-sequence Transformer/README.md`](./Multispectra-to-sequence%20Transformer/README.md)
- [`Plotting/README.md`](./Plotting/README.md)

Model architecture and methodology are described in the manuscript; the subfolder READMEs focus on usage, configuration, and data formats.

## Setup

```bash
conda env create -f environment.yaml
conda activate PolymerSeq
```

## Reproducing results

The three folders are meant to be siblings, as laid out above — outputs from one stage are read as inputs by the next, using relative paths (`../Data Generation/...`, `../Multispectra-to-sequence Transformer/Output/...`).

1. **Generate sequences and spectra** (`Data Generation/`): produce sequence `.csv` files, then simulate spectra into `.h5` datasets, at each noise level and for single-sequence and mixtures data.
2. **Train and evaluate the transformer** (`Multispectra-to-sequence Transformer/`): train one model per spectrum combination and noise level, evaluate on held-out sequences, and run beam search inference — including on spectral mixtures.
3. **Generate figures** (`Plotting/`): produce the paper's figures from the trained models' outputs, plus standalone diagnostic plots for inspecting any individual training run.

See each subfolder's README for exact commands and config file details.

The sequence generation is not seeded, so each call to `Data Generation/seq_generator.py` will generate a different dataset of sequences. The spectra generation is deterministic for UV-Vis and NMR at all noise levels, and is seeded for MS for all random noise added; for the same sequences, the spectra generated should be identical.

The workflow for degeneracy calculations is slightly different - refer to [`Data Generation/README.md`](./Data%20Generation/README.md) for more details.

## Citation

If you use this code or data, please cite the associated manuscript:

**Testing the Limits of Transformer-Based Sequence Inference from Copolymer Spectra**

Additional citation information will be provided following publication.

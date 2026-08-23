# Data Generation

Generates simulated DA copolymer sequences and corresponding UV-Vis, NMR, and mass spectra.
Writes spectra, sequences, and sequence correlations to an HDF5 dataset.

## Folder structure

```text
.
├── configs/ 
│   ├── seq/
│   │    ├── all.yaml
│   │    ├── random.yaml      
│   │    └── mixtures.yaml
│   └── spectra/
│        ├── NOISE0.yaml
│        ├── NOISE1.yaml
│        ├── NOISE2.yaml
│        └── mixtures_NOISE0.yaml
├── degeneracy/                 # contains files to compute spectral degeneracies
├── seq_config.py               # loads a seq_config YAML file into SeqConfig parameter objects
├── seq_generator.py            # reads a seq_config YAML file -> writes .csv file of sequences 
├── spectra_config.py           # loads a spectra_config YAML file into SpectraConfig parameter objects
├── HNMR_dimers.csv             # req. for NMR simulation
├── HNMR_trimers.csv            # req. for NMR simulation
├── spectra_generator.py        # reads a spectra_config YAML file and sequence file -> writes .h5 dataset
├── MS.py                       # Mass spec simulator + parameter classes 
├── NMR.py                      # NMR simulator + parameter classes
└── UV_Vis.py                   # UV-Vis simulator + parameter classes
```

## Sequence generation

### Sequence generator usage

```bash
python seq_generator.py --config configs/seq/all.yaml
python seq_generator.py --config configs/seq/random.yaml
python seq_generator.py --config configs/seq/mixtures.yaml
```

There are three modes for sequence generation: 'all', 'random', and 'mixtures'.

'all' mode: generates one `.csv` file per length in a range of lengths, where each file
contains all possible unique sequences of that length. The files are stored in a directory
specified in the config.

'random' mode: generates a single `.csv` file with randomly generated copolymer sequences,
sampled from a range of lengths using a first-order Markov chain method. The Markov probabilities
are parametrized by a fraction of donor monomers and a sequence correlation (lambda)
which is uniformly sampled from [-1, 1).

Each `.csv` file in 'all' or 'random' modes contains the following columns:

- `sequence` — the copolymer sequence strings
- `lambda` — the corresponding sequence correlation values

'mixtures' mode: generates one `.csv` file per sequence correlation in a range of sequence correlations,
where each file contains a given number of mixtures and the weights in which they should be combined (defined by the 'ratios' variable). The files are stored in a directory specified in the config.

Each `.csv` file in 'mixtures' modes contains the following columns:

- `sequences` — lists of the copolymer sequence strings in each mixture
- `ratios` — lists of the ratios for each copolymer sequence in each mixture
- `lambda` — the corresponding sequence correlation values for each mixture

Note: for each `.csv`, the `ratios` and `lambda` columns have identical values
in each row in this implementation. This is implented so that the code can be modified to accept a different sequence correlation and ratio for each sequence in the mixture.

### Editing a sequence config

'all' mode: edit `configs/seq/all.yaml` to change the range of sequence lengths.

'random' mode: edit `configs/seq/random.yaml` to change the range of sequence lengths,
the fraction of donor monomers, or the number of samples.

'mixtures' mode: edit `configs/seq/mixtures.yaml` to change the range of sequence lengths,
the fraction of donor monomers, the sequence correlations (`lambdas`), the number of sequences per mixture
and their ratios (`ratios`), and the number of mixtures.

## Spectra generation

### Spectra generation usage

```bash
python spectra_generator.py --config configs/spectra/NOISE0.yaml
python spectra_generator.py --config configs/spectra/NOISE1.yaml
python spectra_generator.py --config configs/spectra/NOISE2.yaml

python spectra_generator.py --config configs/spectra/mixtures_NOISE0.yaml
```

There are two modes for spectra generation: 'single' or 'mixtures'; default is 'single'.

'single' mode: generates one `.h5` file with spectra generated from a single sequence.

Each run in 'single' mode produces an `.h5` file with the following datasets:

- `sequence` — the copolymer sequence strings
- `lambda` — the corresponding sequence correlation values
- `uv_vis` / `nmr` / `ms` — spectra arrays, one row per sequence, present
  only for the simulators enabled in the config

'mixtures' mode: generates one `.h5` file with spectra generated from a weighted combination of multiple sequences.

Each run in 'mixtures' mode produces an `.h5` file with the following datasets:

- `sequences` — lists of the copolymer sequence strings in each mixture
- `ratios` — lists of the ratios for each copolymer sequence in each mixture
- `lambda` — the corresponding sequence correlation values for each mixture
- `uv_vis` / `nmr` / `ms` — spectra arrays, one row per mixture, present
  only for the simulators enabled in the config

### Adding a new spectra config

Copy an existing file in `configs/spectra/` and adjust the values you need to
change. A `.csv` file containing sequences must be provided in `Data Generation/`.

Each config file has up to four sections: an overview section plus a parameter
section for the spectra simulators that are turned on under `simulate:`.

```yaml
simulate:
  uv_vis: true
  nmr: true
  ms: true
```

In 'single' mode, the `io:` section specifies the names of the input sequence file and output `.h5` file.

```yaml
io:
  sequence_file: seq_3-20.csv
  output_file: multispectra_NOISE0.h5
```

In 'mixtures' mode, the `io:` section specifies the output directory and lambdas.

```yaml
io: 
  mixtures_lambdas: [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
  output_dir: "mixtures_L10/"
```

| Section | Used When | Feeds Into |
| --- | --- | --- |
| `uv_vis` | `simulate.uv_vis: true` | `FrenkelParameters`, `GaussianPlotParameters` |
| `nmr` | `simulate.nmr: true` | `NMRPlotParameters` |
| `ms` | `simulate.ms: true` | `MassSpecParameters`, `BarPlotParameters`, `MSNoiseParameters` (optional) |

Field-level meaning, units, and defaults for each parameter are documented on the parameter classes themselves in `UV_Vis.py`,
`NMR.py`, and `MS.py`. Note that the NMR simulator requires a trimer file and a dimer (end-group) file, both `.csv`, which contain the
chemical shifts and multiplicities of the atoms in each subgroup.

## Degeneracy Calculation

Details for degeneracy calculations can be found in `./degeneracy/README.md`.

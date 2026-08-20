# Data Generation

Generates simulated DA copolymer sequences and corresponding UV-Vis, NMR, and mass spectra.
Writes spectra, sequences, and sequence correlations to an HDF5 dataset.

## Folder structure

```text
.
├── configs/ 
|   ├── seq_config_all.yaml
|   ├── seq_config_random.yaml                   
│   ├── spectra_config_NOISE0.yaml
│   ├── spectra_config_NOISE1.yaml
|   └── spectra_config_NOISE2.yaml
├── seq_config.py               # loads a seq_config YAML file into SeqConfig parameter objects
├── seq_generator.py            # reads a seq_config YAML file -> writes .csv file of sequences 
├── spectra_config.py           # loads a spectra_config YAML file into SpectraConfig parameter objects
├── input_generator.py          # reads a spectra_config YAML file and sequence file -> writes .h5 dataset
├── MS.py                       # Mass spec simulator + parameter classes 
├── NMR.py                      # NMR simulator + parameter classes
└── UV_Vis.py                   # UV-Vis simulator + parameter classes
```

## Sequence generation

### Sequence generator usage

```bash
python seq_generator.py --config configs/seq_config_all.yaml
python seq_generator.py --config configs/seq_config_random.yaml
```

There are two modes for sequence generation: 'random' and 'all'.

'all' mode: generates one `.csv` file per length in a range of lengths, where each file
contains all possible unique sequences of that length. The files are stored in a directory
specified in the config.

'random' mode: generates a single `.csv` file with randomly generated copolymer sequences,
sampled from a range of lengths using a first-order Markov chain method. The Markov probabilities
are parametrized by a fraction of donor monomers and a sequence correlation (lambda)
which is uniformly sampled from [-1, 1).

Each `.csv` file contains the following columns:

- `sequence` — the copolymer sequence strings
- `lambda` — the corresponding sequence correlation values

### Editing a sequence config

'all' mode: edit `configs/seq_config_all.yaml` to change the range of sequence lengths.

'random' mode: edit `configs/seq_config_random.yaml` to change the range of sequence lengths
or the fraction of donor monomers.

## Spectra generation

### Spectra generation usage

```bash
python input_generator.py --config configs/spectra_config_NOISE0.yaml
python input_generator.py --config configs/spectra_config_NOISE1.yaml
python input_generator.py --config configs/spectra_config_NOISE2.yaml
```

Each run produces an `.h5` file with the following datasets:

- `sequence` — the copolymer sequence strings
- `lambda` — the corresponding sequence correlation values
- `uv_vis` / `nmr` / `ms` — spectra arrays, one row per sequence, present
  only for the simulators enabled in the config

### Adding a new spectra config

Copy an existing file in `configs/` and adjust the values you need to
change. A `.csv` file containing sequences and sequence correlations must be
provided in `Data Generation/`.

Each config file has up to four sections: an overview section plus a parameter
section for the spectra simulators that are turned on under `simulate:`.
The `io:` section specifies the names of the input sequence file and output `.h5` file.

```yaml
simulate:
  uv_vis: true
  nmr: true
  ms: true

io:
  sequence_file: seq_3-20.csv
  output_file: multispectra_NOISE0.h5
```

| Section | Used When | Feeds Into |
| --- | --- | --- |
| `uv_vis` | `simulate.uv_vis: true` | `FrenkelParameters`, `GaussianPlotParameters` |
| `nmr` | `simulate.nmr: true` | `NMRPlotParameters` |
| `ms` | `simulate.ms: true` | `MassSpecParameters`, `BarPlotParameters`, `MSNoiseParameters` (optional) |

Field-level meaning, units, and defaults for each parameter are documented on the parameter classes themselves in `UV_Vis.py`,
`NMR.py`, and `MS.py`. Note that the NMR simulator requires a trimer file and a dimer (end-group) file, both `.csv`, which contain the
chemical shifts and multiplicities of the atoms in each subgroup.

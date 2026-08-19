# Data Generation

Generates simulated DA copolymer sequences and corresponding UV-Vis, NMR, and mass spectra.
Writes spectra, sequences, and sequence correlations to an HDF5 dataset.

## Project structure

```text
.
├── configs/                    # one YAML file per noise level
│   ├── spectra_config_NOISE0.yaml
│   └── spectra_config_NOISE1.yaml
|   └── spectra_config_NOISE2.yaml
├── spectra_config.py           # loads a YAML file into SpectraConfig parameter objects
├── input_generator.py          # reads config and sequence file -> writes .h5 dataset
├── MS.py                       # Mass spec simulator + parameter classes 
├── NMR.py                      # NMR simulator + parameter classes
└── UV_Vis.py                   # UV-Vis simulator + parameter classes
```

## Spectra generation

### Usage

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

### Adding a new config

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

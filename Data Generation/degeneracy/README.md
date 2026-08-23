# Degeneracy Calculation

Quantifies how often distinct copolymer sequences produce indistinguishable (degenerate) spectra for a given modality over a defined copolymer sequence space. For a dataset generated from copolymers in this space, calculates the portion of the dataset that have spectral degeneracies within the space and, based on the degeneracy, computes an expected reconstruction accuracy on the test set for comparison with model reconstruction.

`degeneracy_functions.py` contains modality-agnostic machinery. In addition, included are scripts to compute degeneracies for each modality (MS, NMR, UV-Vis). Each script defines a `getDiscreteSpectrum` function and an `isIndistinguishabe` function, picks an appropriate comparison strategy, and calls the functions defined in `degeneracy_functions.py` to calculate the degeneracy metrics.

## Comparison strategies

Two ways to decide whether two sequences' spectra are indistinguishable —
pick only one for each modality.

| | `exact_match_degeneracy` | `threshold_match_degeneracy` |
| --- | --- | --- |
| Use for | MS, NMR — naturally discrete/quantized signals (integer fragment masses, integer trimer counts) | UV-Vis — continuous transition energies/intensities, where exact matching after rounding can miss near-duplicates right at rounding boundaries |
| Matching | exact equality | pairwise threshold; checks whether each `x` and `intensity` difference between two spectra falls within `x_res`/`intensity_res` |
| `getDiscreteSpectrum` signature | `getDiscreteSpectrum(seq) -> tuple` | `getDiscreteSpectrum(seq, max_length) -> (x, intensity)`, each pre-padded to a common length |

Both return a `DataFrame['Sequence', 'Degeneracy']` with one row per input
sequence. `Degeneracy` is how many sequences (including itself) share that sequence's spectrum.
`Degeneracy == 1` means the sequence has a unique spectrum.

`threshold_match_degeneracy` can be computed on a GPU: it compares `N` sequences by doing an `N×N` pairwise
comparison in `batch_size`-sized chunks.

## Pipeline

The three stages run in this order, each depending on the previous stage's
output files:

1. **`all_lengths_degeneracy(getDiscreteSpectrum, degeneracy_fn, out_folder, min_length, max_length, sequence_dir)`**
   Exhaustive analysis: for every length in `[min_length, max_length]`, reads
   `{sequence_dir}/all_seq_{length}.csv` (every possible sequence of that
   length) and compares all of them against each other using `degeneracy_fn`
   (one of the two strategies above). Writes `{out_folder}/degeneracies_{length}.csv`
   per length, plus an `{out_folder}/degeneracy.csv` summary (contains the number of unique-spectra and size of the largest degenerate group, at each length).
2. **`dataset_degeneracies(out_folder, min_length, max_length, sequence_file)`**
   Looks up every sequence in the *actual* dataset (`sequence_file` — e.g.
   `seq_3-20.csv`) against the exhaustive reference computed in step 1 (via
   `get_degeneracy_map`, which merges all the per-length files into one
   `{sequence: degeneracy}` map). A dataset sequence not found in that map
   defaults to `Degeneracy = 1` — this should only happen if `sequence_file`
   contains a length outside `[min_length, max_length]`, or step 1 hasn't
   been run for the full range yet.
   Splits the dataset into train/validate/test using `stratified_split_indices`
   from `Multispectra-to-sequence Transformer/SequenceEncoder.py` — the same
   split logic actual model training uses, so the reported degeneracy
   statistics apply to the same train/val/test sets the model sees. Writes,
   under `{out_folder}/dataset/`: per-split sequence+degeneracy CSVs,
   per-split degeneracy-distribution CSVs, and a `dataset_degeneracies.txt`
   summary — number of sequence, number of unique-spectra, and largest degenerate
   group, and an *expected* reconstruction accuracy per split (weighted sum
   of `1/degeneracy` across all sequences — the accuracy a model could
   achieve even if every prediction within a degenerate group were "correct"
   for that group).
3. **`degeneracy_aware_reconstruction(isIndistinguishable, model_out_folder)`**
   Re-scores a trained model's actual prediction errors: reads
   `{model_out_folder}/errors.csv` (must have `target sequence` /
   `predicted sequence` columns), flags each mispredicted pair as `Degenerate`
   if `isIndistinguishable(target, predicted)` is `True` — i.e. the model's
   "wrong" answer is nonetheless spectrally indistinguishable from the right
   one. Writes `{model_out_folder}/degenerate_errors.csv` (the original
   errors plus the `Degenerate` column) and `degeneracy_aware_accuracy.txt` (recomputed correct/total based on degeneracy).
   `isIndistinguishable(seq_a, seq_b) -> bool` is supplied by the modality
   script`.

Steps 1 & 2 need only be performed once over the copolymer space, and must be computed relative to the noise-free (NOISE0) spectra. Step 3 can be recomputed anytime a different subset of sequences is used, if different train/val/test split is used, or if the model is retrained. We compute `degeneracy_aware_reconstruction` for the a single model's predictions when trained on NOISE0 spectra.

## Cross-repo dependency

Both `degeneracy_functions.py` and the modality scripts require files located in other directories.

To generate spectra for comparisons, each spectral modality script needs access to the NOISE0 spectra configurations (located at `Data Generation/configs/spectra/NOISE0.yaml`), as well as the spectral simulators located in `Data Generation`.

To compute the degeneracies at each length of the copolymer sequence space, `degeneracy_functions.py` requires the `all_seq_{length}.csv` files previously generated within `Data Generation/all/`.

To compute the train/val/test degeneracies, `degeneracy_functions.py` requires the same `stratified_split_indices()` function used to generate the splits used in the models; this function is defined in `Multispectra-to-sequence Transformer/SequenceEncoder.py`. In addition, each spectral modality file needs to pass in `seq_3-20.csv` that contains the copolymer sequences in the dataset, which is located in `Data Generation/`.

Finally, to compute degeneracy-aware reconstruction, each spectral modality file needs the predictions from a trained model, which are located in `{modality}/NOISE0/errors/errors.csv`.

The expected locations of each cross-repo dependency are computed in the respective files. For clarity, a visual depiction of the relevant folder structure is illustrated below.

### Relevant folder structure

```text
.
├── Data Generation
│   ├── all/                        # contains all possible sequences at lengths 3-20
│   │   ├── all_seq_3.csv
│   │   └── all_seq_20.csv
│   ├── configs/ 
│   │   ├── seq/
│   │   └── spectra/
│   │       └── NOISE0.yaml
│   ├── degeneracy/                 
│   │   ├── degeneracy_functions.py
│   │   ├── ms_degeneracy.py
│   │   ├── nmr_degeneracy.py
│   │   └── uv_vis_degeneracy.py
│   ├── seq_3-20.csv                # copolymer sequence used in the dataset
│   ├── MS.py                       # Mass spec simulator + parameter classes 
│   ├── NMR.py                      # NMR simulator + parameter classes
│   └── UV_Vis.py                   # UV-Vis simulator + parameter classes
└── Multispectra-to-sequence Transformer
    ├── Output/
    │   └── ms/
    │       └── NOISE0/
    │           └── errors/
    │               └── errors.csv  # error file for a model run         
    └── SequenceEncoder.py          # contains stratified_split_indices function
```

## Adding a new modality

Three modality scripts are included (`ms_degeneracy.py`, `nmr_degeneracy.py`, `uv_vis_degeneracy.py`) that can be used for reference.

A modality script needs to:

```python
def getDiscreteX(sequence, max_length=None):
    ...  # exact_match: getDiscreteX(seq) -> tuple
         # threshold_match: getDiscreteX(seq, max_length) -> (x, intensity)
 
def isIndistinguishable(seq_a, seq_b):
    ...  # compare two sequences' spectra directly, however this modality needs to
 
def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    all_lengths_degeneracy(getDiscreteX, exact_match_degeneracy,  # or threshold_match_degeneracy
                            OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_DIR)
    dataset_degeneracies(OUT_FOLDER, MIN_LENGTH, MAX_LENGTH, SEQUENCE_FILE)
    degeneracy_aware_reconstruction(isIndistinguishable, MODEL_OUTPUT_FOLDER)
```

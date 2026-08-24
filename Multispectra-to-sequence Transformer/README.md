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

"""
Config loader/writer for multispectra-to-sequence transformer
"""

from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class RunConfig:
    """Parameters for training and evaluation of single-sequence spectra"""
    data_file: str
    output_dir: str
    saved_model: bool   # if True, load saved model from output_dir

    # dictionary containing the spectra to include and whether to scale them (True/False)
    scale: dict         

    # Transformer parameters
    d_model: int
    h: int
    N: int
    d_ff: int
    dropout: float

    # Training parameters
    epochs: int
    batch_size: int

    # Beam search parameters
    beam_width: int
    alpha: float

@dataclass
class RunMixturesConfig:
    """Parameters for running beam search inference on mixtures data using saved model."""
    mixtures_file_template: str
    mixtures_lambdas: list
    output_dir: str

    # dictionary containing the spectra to include and whether to scale them (True/False)
    scale: dict         

    # Beam search parameters
    beam_width: int
    alpha: float

@dataclass
class ModelArchitecture:
    """Parameters for the model architecture"""
    spec_lengths: dict
    seq_length: int
    vocab_size: int

    d_model: int
    h: int
    N: int
    d_ff: int
    dropout: float

def load_config(path: str) -> RunConfig:
    """Load configuration from a YAML file and return a RunConfig object"""

    raw = yaml.safe_load(Path(path).read_text())

    io = raw["io"]
    model = raw["model"]
    training = raw["training"]
    beam_search = raw["beam_search"]

    return RunConfig(
        data_file=io["data_file"],
        output_dir=io["output_dir"],
        saved_model=io["saved_model"],

        scale=raw["scale"],

        d_model=model["d_model"],
        h=model["h"],
        N=model["N"],
        d_ff=model["d_ff"],
        dropout=model["dropout"],

        epochs=training["epochs"],
        batch_size=training["batch_size"],

        beam_width=beam_search["beam_width"],
        alpha=beam_search["alpha"]
    )

def load_mix_config(path: str) -> RunMixturesConfig:
    """Load configuration from a YAML file and return a RunMixturesConfig object"""

    raw = yaml.safe_load(Path(path).read_text())

    io = raw["io"]
    beam_search = raw["beam_search"]

    return RunMixturesConfig(
        mixtures_file_template=io["mixtures_file_template"],
        mixtures_lambdas=io["mixtures_lambdas"],
        output_dir=io["output_dir"],

        scale=raw["scale"],

        beam_width=beam_search["beam_width"],
        alpha=beam_search["alpha"]
    )

def write_model_architecture(path: str, config_dict: dict):
    """Write a ModelArchitecture object to a YAML file"""

    with open(path, 'w') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

def load_model_architecture(path: str) -> ModelArchitecture:
    """Load model architecture from a YAML file and return a ModelArchitecture object"""

    raw = yaml.safe_load(Path(path).read_text())

    return ModelArchitecture(
        spec_lengths=raw["spec_lengths"],
        seq_length=raw["seq_length"],
        vocab_size=raw["vocab_size"],

        d_model=raw["d_model"],
        h=raw["h"],
        N=raw["N"],
        d_ff=raw["d_ff"],
        dropout=raw["dropout"]
    )

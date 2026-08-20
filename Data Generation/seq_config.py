"""
Config loader for copolymer sequence generation
"""

from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class SeqConfig:
    """Parameters for copolymer sequence generation"""
    mode: str           # 'random' or 'all'
    monomers: list

    min_length: int = None
    max_length: int = None

    # mode: random
    f_D: float = None
    num_sequences: int = None

    # mode: all
    output_dir: str = None

def load_config(path: str) -> SeqConfig:
    raw = yaml.safe_load(Path(path).read_text())

    mode = raw["mode"]
    monomers = raw["monomers"]

    min_length = raw["min_length"]
    max_length = raw["max_length"]

    cfg = SeqConfig(
        mode = mode,
        monomers = monomers,

        min_length = min_length,
        max_length = max_length
    )

    if mode == "random":
        cfg.f_D = raw["random"]["f_D"]
        cfg.num_sequences = raw["random"]["num_sequences"]

    elif mode =="all":
        cfg.output_dir = raw["all"]["output_dir"]

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'random' or 'all'.")
    
    return cfg

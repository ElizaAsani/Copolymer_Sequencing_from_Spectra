"""
Config loader for copolymer spectral simulation
"""

from dataclasses import dataclass
from pathlib import Path
import yaml

from UV_Vis import UV_Vis, FrenkelParameters, GaussianPlotParameters
from NMR import NMRPlotParameters
from MS import MSParameters, BarPlotParameters, MSNoiseParameters

@dataclass
class SpectraConfig:
    """Parameters for copolymer spectral simulation"""
    simulate_uv_vis: bool
    simulate_nmr: bool
    simulate_ms: bool

    sequence_file: str
    output_file: str

    monomers: list

    frenkel_params: "FrenkelParameters | None" = None
    uv_vis_plot_params: "GaussianPlotParameters | None" = None

    trimer_file: "str | None" = None
    dimer_file: "str | None" = None
    nmr_plot_params: "NMRPlotParameters | None" = None

    ms_params: "MSParameters | None" = None
    ms_plot_params: "BarPlotParameters | None" = None
    ms_noise_params: "MSNoiseParameters | None" = None

def load_config(path: str) -> SpectraConfig:
    raw = yaml.safe_load(Path(path).read_text())

    monomers = raw["monomers"]
    simulate = raw["simulate"]
    io = raw["io"]

    cfg = SpectraConfig(
        simulate_uv_vis = simulate["uv_vis"],
        simulate_nmr = simulate["nmr"],
        simulate_ms = simulate["ms"],

        sequence_file = io["sequence_file"],
        output_file = io["output_file"],

        monomers = monomers
    )

    if cfg.simulate_uv_vis:
        uv = raw["uv_vis"]
        if "J_DD" in uv["dipole_dipole"]:
            J_DD = uv["dipole_dipole"]["J_DD"]
        else:
            J_DD = UV_Vis.calculateJDD(uv["dipole_dipole"]["R"], uv["dipole_dipole"]["eps_r"])

        cfg.frenkel_params = FrenkelParameters(
            monomers = monomers,
            eps = uv["frenkel"]["eps"],
            mu = uv["frenkel"]["mu"],
            J_DD = J_DD,
            J_SE = uv["frenkel"]["J_SE"]
        )

        cfg.uv_vis_plot_params = GaussianPlotParameters(
            points = uv["plot"]["points"],
            std_dev = uv["plot"]["std_dev"],
            energy_range = uv["plot"].get("energy_range", None),
            wavelength_range = uv["plot"].get("wavelength_range", None),
            peaks = uv["plot"].get("peaks", None)
        )

    if cfg.simulate_nmr:
        nmr = raw["nmr"]

        cfg.trimer_file = nmr["trimer_file"]
        cfg.dimer_file = nmr["dimer_file"]

        cfg.nmr_plot_params = NMRPlotParameters(
            points = nmr["plot"]["points"],
            half_width = nmr["plot"]["half_width"],
            shift_range = nmr["plot"]["shift_range"],
            reference_shift = nmr["plot"]["reference_shift"],
            tolerance = nmr["plot"].get("tolerance", NMRPlotParameters.tolerance)
        )

    if cfg.simulate_ms:
        ms = raw["ms"]

        cfg.ms_params = MSParameters(
            monomers = monomers + ["*"],
            formulas = ms["formulas"],
            s = ms.get("s", MSParameters.s)
        )

        cfg.ms_plot_params = BarPlotParameters(
            mass_range = ms["plot"]["mass_range"],
            bin_width = ms["plot"]["bin_width"]
        )

        if "noise" in ms:
            cfg.ms_noise_params = MSNoiseParameters(
                dropout = ms["noise"].get("dropout", MSNoiseParameters.dropout),
                extra_peaks = ms["noise"].get("extra_peaks", MSNoiseParameters.extra_peaks),
                width = ms["noise"].get("width", MSNoiseParameters.width),
                weight = ms["noise"].get("weight", MSNoiseParameters.weight)
            )
        else:
            cfg.ms_noise_params = MSNoiseParameters()  

    return cfg

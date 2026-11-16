"""
ccsn_ml
=======

Clean, modular implementation of CCSN gravitational-wave classification.

This package is designed to:
- Start from raw waveform `.txt` files (one time series per file).
- Support both 1D (time-series) and 2D (spectrogram) CNN pipelines.
- Handle multiple class groupings and SNR ranges.
- Run both locally and on clusters (e.g. via Condor) using configuration files.
"""

__all__ = [
    "data",
    "datasets",
    "models",
    "train",
    "evaluate",
    "features",
    "calibration",
    "groupings",
]


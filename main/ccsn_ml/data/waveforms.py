"""
Loading raw CCSN waveform time series from `.txt` files.

Assumptions (aligned with your dissertation and notes):
- One waveform per file.
- Single column of ~16k amplitude samples (4 seconds at 4096 Hz).
- Files are grouped at least by simulation family (paper), and possibly by mechanism.

This module will be extended to:
- Discover files based on a metadata table or directory convention.
- Return waveforms along with labels (mechanism, family) and SNR-related metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Every CCSN waveform in the dissertation is 4 seconds at 4096 Hz.
FS_HZ = 4096
DURATION_S = 4
WAVEFORM_SAMPLES = FS_HZ * DURATION_S  # 16384


def require_waveform_length(length: int) -> None:
    """Reject any series that is not the full 4-second waveform."""
    if int(length) != WAVEFORM_SAMPLES:
        raise ValueError(
            f"Expected {WAVEFORM_SAMPLES} samples ({DURATION_S} s at {FS_HZ} Hz), got {length}."
        )


def stable_seed(base: int, key: str) -> int:
    """Process-independent seed. Python's hash() is not stable across runs."""
    digest = hashlib.md5(f"{base}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def allocate_noise_counts(
    split_names: Sequence[str],
    ratios: Sequence[float],
    noise_total: int,
    noise_counts: Dict[str, int] | None,
) -> Dict[str, int]:
    """
    How many synthetic Gaussian noise waveforms each split gets.

    Explicit counts match the original scripts:
    - 1D grouped: Dir_main/100k/1D_CNN/datasets/add.py uses train/val/test = 2000/600/600
    - 2D spectrograms: noise_specs_add.py uses train/val/test = 3000/600/600
    """
    names = list(split_names)
    if noise_counts is not None:
        unknown = set(noise_counts) - set(names)
        if unknown:
            raise ValueError(f"Unknown noise split names: {sorted(unknown)}")
        return {name: int(noise_counts.get(name, 0)) for name in names}

    if noise_total <= 0:
        return {name: 0 for name in names}

    ratio_list = [float(r) for r in ratios]
    total = sum(ratio_list)
    ratio_list = [r / total for r in ratio_list]
    raw = [int(round(noise_total * r)) for r in ratio_list]
    # Keep the sum equal to noise_total by adjusting the last split.
    raw[-1] = noise_total - sum(raw[:-1])
    if raw[-1] < 0:
        raise ValueError(f"Noise split ratios over-allocated {noise_total} samples: {raw}")
    return {name: count for name, count in zip(names, raw)}


@dataclass
class WaveformRecord:
    """Container for a single waveform and its metadata."""

    path: Path
    samples: np.ndarray  # shape: (N,)
    mechanism: Optional[str] = None
    family: Optional[str] = None  # e.g. "radice2019"
    snr: Optional[float] = None
    # Additional fields (amplitude, phase, etc.) can be added when we parse log files.


def load_waveform_txt(path: Path) -> np.ndarray:
    """Load a single-column waveform from a `.txt` file."""
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError(f"Expected 1D waveform in {path}, got shape {data.shape}")
    return data


def iter_waveforms(
    root: Path,
    pattern: str = "**/*.txt",
) -> Iterable[WaveformRecord]:
    """
    Yield WaveformRecord objects for all `.txt` files under `root`.

    For now this only fills `path` and `samples`. Mechanism/family/SNR
    will be attached later once we formalize the directory and log-file
    conventions.
    """
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        samples = load_waveform_txt(p)
        yield WaveformRecord(path=p, samples=samples)


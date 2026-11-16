"""
Preprocessing utilities for CCSN waveforms.

Current scope (paper-consistent 2D spectrogram recipe):
- STFT/spectrogram with:
  - sampling rate `fs=4096`
  - window: Kaiser with beta=5.6
  - segment length / nfft: 512
  - overlap: 0.75
- Spectrogram post-processing:
  - log1p scaling
  - normalize to [0, 1]
  - histogram equalization (OpenCV `equalizeHist`)
  - resize to 256x256
  - render to an RGB PNG-compatible representation (3 channels)

Later scope:
- Whitening / normalization (ensure we do not double-normalize if your raw txts are already normalized)
- SNR association (parsing SNR from per-sample logs and keeping mapping consistent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import spectrogram
from numpy import kaiser
from scipy.ndimage import zoom

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass
class SpectrogramConfig:
    """Configuration for the paper-consistent spectrogram pipeline."""

    # Spectrogram parameters
    fs: float = 4096.0
    nfft: int = 512
    overlap: float = 0.75
    window_beta: float = 5.6  # Kaiser beta, per dissertation

    # Post-processing
    log1p: bool = True
    equalize_hist: bool = True
    resize: Tuple[int, int] = (256, 256)
    cmap: str = "viridis"  # matches your preprocessing script
    origin: str = "lower"  # matches imshow(origin="lower")

    # Numerical stability
    eps: float = 1e-12


def whiten_and_normalize(x: np.ndarray) -> np.ndarray:
    """
    Placeholder for whitening and normalization, to be made consistent
    with the implementation used in your original pipeline.
    """
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    return x


def _enhance_sxx(sxx: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    """
    Copy of your contrast-enhancement logic (from `mech_to_spec_high_c.py` / `noise_specs_add.py`):
    - log1p
    - normalize to [0, 1]
    - scale to uint8 [0,255]
    - histogram equalization
    """
    if cfg.log1p:
        sxx = np.log1p(sxx)

    s_min = float(np.min(sxx))
    s_max = float(np.max(sxx))
    s_range = s_max - s_min
    if s_range < cfg.eps:
        sxx_norm = np.zeros_like(sxx, dtype=np.float32)
    else:
        sxx_norm = (sxx - s_min) / (s_range + cfg.eps)

    if cfg.equalize_hist:
        sxx_u8 = (sxx_norm * 255.0).astype(np.uint8)
        sxx_eq = cv2.equalizeHist(sxx_u8)
        return sxx_eq

    # If equalization is disabled, return float in [0,1]
    return sxx_norm.astype(np.float32)


def compute_spectrogram_sxx(
    x: np.ndarray,
    cfg: SpectrogramConfig,
) -> np.ndarray:
    """
    Compute the raw spectrogram power `Sxx` using `scipy.signal.spectrogram`.
    Returns a 2D array Sxx (frequency bins x time bins).
    """
    noverlap = int(cfg.nfft * cfg.overlap)
    _, _, sxx = spectrogram(
        x,
        fs=cfg.fs,
        nperseg=cfg.nfft,
        noverlap=noverlap,
        nfft=cfg.nfft,
        detrend="constant",
        window=kaiser(cfg.nfft, cfg.window_beta),
        scaling="density",
        mode="psd",
    )
    return sxx.astype(np.float32)


def spectrogram_to_rgb(
    x: np.ndarray,
    cfg: SpectrogramConfig,
) -> np.ndarray:
    """
    Full paper-consistent pipeline for a single waveform:
    waveform -> Sxx -> enhancement -> resize -> colormap -> RGB array.

    Returns:
      RGB uint8 array with shape (resize[1], resize[0], 3) to match (H, W, C).
    """
    sxx = compute_spectrogram_sxx(x, cfg)
    sxx_enhanced = _enhance_sxx(sxx, cfg)

    # Resize Sxx to cfg.resize (height, width)
    zoom_factors = (
        cfg.resize[1] / sxx_enhanced.shape[0],  # rows (frequency bins)
        cfg.resize[0] / sxx_enhanced.shape[1],  # cols (time bins)
    )
    sxx_resized = zoom(sxx_enhanced, zoom_factors, order=3)

    # Match imshow/colormap normalization behavior by normalizing to min/max
    vmin = float(np.min(sxx_resized))
    vmax = float(np.max(sxx_resized))
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cfg.cmap)
    rgba = cmap(norm(sxx_resized))  # (H, W, 4) float in [0,1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

    return rgb


def save_rgb_image(rgb: np.ndarray, out_path: str | "os.PathLike[str]") -> None:
    """Save an RGB uint8 array as a PNG."""
    import os
    from PIL import Image

    out_dir = os.path.dirname(str(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(out_path)



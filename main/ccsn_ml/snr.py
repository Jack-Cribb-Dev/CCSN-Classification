"""
SNR utilities.

This module migrates the logic from `Dir_main/100k/FINALS/utils/snr_mapper.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


def load_snr_values(snr_dir: str | Path) -> Dict[str, List[float]]:
    """
    Load all `*_snrs.txt` files under `snr_dir`.

    Expected format (mirrors your FINALS pipeline):
    - filenames like: `andresen2019_snrs.txt` (or `andresen2019_snrs-2.txt`)
    - file contents: one float per line
    """
    snr_dir = Path(snr_dir)
    snr_dict: Dict[str, List[float]] = {}
    if not snr_dir.exists():
        raise FileNotFoundError(f"Missing snr_dir: {snr_dir}")

    for fname in os.listdir(snr_dir):
        if not fname.endswith(".txt"):
            continue
        # Support filenames like:
        # - andresen2019_snrs.txt
        # - andresen2019_snrs-2.txt
        # - andresen2019_snrs_anything.txt
        if "snrs" not in fname:
            continue

        base = fname[: -len(".txt")]
        if "_snrs" in base:
            model = base.split("_snrs", 1)[0]
        else:
            # Fallback: take prefix before first occurrence of "snrs"
            model = base.split("snrs", 1)[0].rstrip("-_")

        with open(snr_dir / fname, "r", encoding="utf-8") as f:
            snrs = [float(line.strip()) for line in f if line.strip()]
        snr_dict[model] = snrs

    return snr_dict


def extract_snr_from_filename(image_path: str | Path, snr_dict: Dict[str, List[float]]) -> Optional[float]:
    """
    Extract (model, index) from an image filename and return the matching SNR.

    Mirrors your FINALS assumption:
    - image filename format: `<model>_<index>.png`
    - SNR index corresponds to line index inside `<model>_snrs.txt`
    """
    image_path = Path(image_path)
    name_part = image_path.stem  # removes suffix (e.g. ".png")
    try:
        model, index_s = name_part.rsplit("_", 1)
        index = int(index_s)
        return snr_dict[model][index]
    except (ValueError, KeyError, IndexError):
        return None


"""
Build 2D spectrogram datasets from raw waveform `.txt` files.

This is the missing bridge between:
- raw waveforms + (optionally) per-sample log files (for SNR)
- paper-consistent spectrogram generation
- directory layout compatible with `tf.keras.utils.image_dataset_from_directory`

Important: In the repository your final model training currently consumes
pre-generated image directories. This builder makes it possible to generate
those directories from raw waveforms in a reproducible way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .preprocess import SpectrogramConfig, spectrogram_to_rgb, save_rgb_image
from .waveforms import (
    WAVEFORM_SAMPLES,
    allocate_noise_counts,
    load_waveform_txt,
    require_waveform_length,
    stable_seed,
)
from ..snr import extract_snr_from_filename, load_snr_values


@dataclass(frozen=True)
class DatasetSplit:
    name: str  # "train" / "val" / "test"
    ratio: float


@dataclass
class SpectrogramDatasetBuilderConfig:
    """
    Builder configuration for 2D dataset generation.

    Assumes:
    - raw_root/<mechanism_label>/*.txt are waveform files
    - mechanism_label is the class label for 20-class classification,
      or is mapped to a group label for grouped classification.
    """

    raw_root: Path
    output_base: Path  # e.g. base_path/datasets/A
    dataset_variant: str  # e.g. "A" (for datasets/A)
    experiment_name: str  # e.g. "A2" -> Train_A2/Val_A2/Test_A2

    mechanism_to_class: Dict[str, str]  # mechanism folder name -> output class folder name
    classes: List[str]  # output class folder names in desired label space

    splits: Tuple[DatasetSplit, DatasetSplit, DatasetSplit] = (
        DatasetSplit("train", 0.7),
        DatasetSplit("val", 0.15),
        DatasetSplit("test", 0.15),
    )
    split_seed: int = 42

    spectrogram_cfg: SpectrogramConfig = SpectrogramConfig()

    # Optional noise class. Explicit counts match noise_specs_add.py (3000/600/600).
    # noise_total is used only when noise_counts is omitted.
    noise_total: int = 0
    noise_counts: Dict[str, int] | None = None
    noise_class_label: str = "noise"

    # Optional A2-style cut from generate_A2.py (keep SNR >= snr_min).
    snr_min: float | None = None
    snr_dir: Path | None = None


def _collect_waveform_paths(raw_root: Path, mechanism_to_class: Dict[str, str]) -> Dict[str, List[Path]]:
    """Collect waveform paths per mechanism folder name."""
    mech_to_paths: Dict[str, List[Path]] = {m: [] for m in mechanism_to_class.keys()}
    for mechanism in mech_to_paths.keys():
        mech_dir = raw_root / mechanism
        if not mech_dir.exists():
            # Your "noise" class is synthetic in the paper. If raw waveforms don't include
            # a `noise/` directory, we can still generate it via `noise_total`.
            if str(mechanism).lower() == "noise":
                mech_to_paths[mechanism] = []
                continue
            raise FileNotFoundError(f"Missing mechanism folder: {mech_dir}")
        for p in mech_dir.glob("*.txt"):
            if p.is_file():
                mech_to_paths[mechanism].append(p)
        if not mech_to_paths[mechanism]:
            if str(mechanism).lower() == "noise":
                # Again: noise can be generated synthetically
                mech_to_paths[mechanism] = []
                continue
            raise FileNotFoundError(f"No .txt files found under {mech_dir}")
    return mech_to_paths


def _split_paths(paths: List[Path], splits: Tuple[DatasetSplit, DatasetSplit, DatasetSplit], seed: int) -> Dict[str, List[Path]]:
    """Deterministically split paths into train/val/test subsets."""
    rng = np.random.default_rng(seed)
    paths_sorted = list(paths)
    rng.shuffle(paths_sorted)

    ratios = [s.ratio for s in splits]
    total = sum(ratios)
    ratios = [r / total for r in ratios]

    n = len(paths_sorted)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # Ensure test gets the remainder
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    out: Dict[str, List[Path]] = {}
    out[splits[0].name] = paths_sorted[:n_train]
    out[splits[1].name] = paths_sorted[n_train:n_train + n_val]
    out[splits[2].name] = paths_sorted[n_train + n_val:n_train + n_val + n_test]
    return out


def build_2d_spectrogram_dataset(cfg: SpectrogramDatasetBuilderConfig) -> None:
    """
    Generate RGB spectrogram PNGs into:
    - output_base/Train_<experiment_name>/<class>/*.png
    - output_base/Val_<experiment_name>/<class>/*.png
    - output_base/Test_<experiment_name>/<class>/*.png
    """
    # Output directories (match your existing FINALS directory naming)
    train_out = cfg.output_base / cfg.dataset_variant / f"Train_{cfg.experiment_name}"
    val_out = cfg.output_base / cfg.dataset_variant / f"Val_{cfg.experiment_name}"
    test_out = cfg.output_base / cfg.dataset_variant / f"Test_{cfg.experiment_name}"

    split_to_out = {"train": train_out, "val": val_out, "test": test_out}

    for class_label in cfg.classes:
        for out_dir in split_to_out.values():
            (out_dir / class_label).mkdir(parents=True, exist_ok=True)

    # Collect raw waveforms
    mech_to_paths = _collect_waveform_paths(cfg.raw_root, cfg.mechanism_to_class)
    snr_dict = None
    if cfg.snr_min is not None:
        if cfg.snr_dir is None:
            raise ValueError("snr_min requires snr_dir (directory of *_snrs*.txt files).")
        snr_dict = load_snr_values(cfg.snr_dir)

    # Split + generate
    for mechanism, paths in mech_to_paths.items():
        class_label = cfg.mechanism_to_class[mechanism]
        split_paths = _split_paths(paths, cfg.splits, seed=stable_seed(cfg.split_seed, mechanism))

        for split_name, split_list in split_paths.items():
            out_dir = split_to_out[split_name] / class_label
            for p in tqdm(split_list, desc=f"{mechanism}->{class_label} ({split_name})"):
                if snr_dict is not None:
                    snr = extract_snr_from_filename(p, snr_dict)
                    if snr is None or snr < cfg.snr_min:
                        continue
                x = load_waveform_txt(p)
                require_waveform_length(x.shape[0])
                rgb = spectrogram_to_rgb(x, cfg.spectrogram_cfg)
                out_path = out_dir / (p.stem + ".png")
                save_rgb_image(rgb, out_path)

    # Optional noise class. Length is always the full 4-second waveform.
    split_names = [s.name for s in cfg.splits]
    counts = allocate_noise_counts(
        split_names,
        [s.ratio for s in cfg.splits],
        cfg.noise_total,
        cfg.noise_counts,
    )
    if any(counts.values()):
        noise_class = cfg.noise_class_label
        for split_name, n_noise in counts.items():
            out_dir = split_to_out[split_name] / noise_class
            out_dir.mkdir(parents=True, exist_ok=True)
            noise_rng = np.random.default_rng(stable_seed(cfg.split_seed, f"noise:{split_name}"))
            for i in tqdm(range(n_noise), desc=f"noise ({split_name})"):
                noise = noise_rng.normal(0.0, 1.0, size=(WAVEFORM_SAMPLES,)).astype(np.float32)
                rgb = spectrogram_to_rgb(noise, cfg.spectrogram_cfg)
                save_rgb_image(rgb, out_dir / f"noise_{i:05d}.png")


"""
Build 1D waveform datasets from raw waveform `.txt` files.

This supports both:
- Full mechanism classification (20 classes)
- Grouped classification (e.g. 9 groups)

The implementation mirrors your existing 1D preprocessing logic from `Dir_main/100k/1D_CNN/datasets/gen_grouped.py`:
- Each waveform is normalized per-sample using (x - mean) / std.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
    name: str
    ratio: float


@dataclass
class WaveformDatasetBuilderConfig:
    raw_root: Path  # contains mechanism subfolders
    mechanism_to_class: Dict[str, str]  # mechanism folder -> class label folder name (string)
    classes: List[str]  # output class labels in desired label index space

    input_length: int = WAVEFORM_SAMPLES
    normalize_per_sample: bool = True

    splits: Tuple[DatasetSplit, DatasetSplit, DatasetSplit] = (
        DatasetSplit("train", 0.7),
        DatasetSplit("val", 0.15),
        DatasetSplit("test", 0.15),
    )
    split_seed: int = 42

    noise_total: int = 0  # used only when noise_counts is omitted
    noise_counts: Dict[str, int] | None = None  # e.g. {"train": 2000, "val": 600, "test": 600}
    noise_class_label: str = "noise"

    # generate_A2.py keeps a sample only when its logged SNR is at least this value.
    snr_min: float | None = None
    snr_dir: Path | None = None


def _collect_waveform_paths(raw_root: Path, mechanisms: List[str]) -> Dict[str, List[Path]]:
    mech_to_paths: Dict[str, List[Path]] = {m: [] for m in mechanisms}
    for mech in mechanisms:
        mech_dir = raw_root / mech
        if not mech_dir.exists():
            # Your "noise" class is synthetic in the paper. If raw waveforms don't include it,
            # we will generate noise samples later using cfg.noise_total.
            if str(mech).lower() == "noise":
                mech_to_paths[mech] = []
                continue
            raise FileNotFoundError(f"Missing mechanism folder: {mech_dir}")
        mech_to_paths[mech] = sorted([p for p in mech_dir.glob("*.txt") if p.is_file()])
        if not mech_to_paths[mech]:
            if str(mech).lower() == "noise":
                mech_to_paths[mech] = []
                continue
            raise FileNotFoundError(f"No .txt waveforms found under {mech_dir}")
    return mech_to_paths


def _split_paths(paths: List[Path], splits: Tuple[DatasetSplit, DatasetSplit, DatasetSplit], seed: int) -> Dict[str, List[Path]]:
    rng = np.random.default_rng(seed)
    paths_copy = list(paths)
    rng.shuffle(paths_copy)

    ratios = [s.ratio for s in splits]
    total = sum(ratios)
    ratios = [r / total for r in ratios]

    n = len(paths_copy)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    return {
        splits[0].name: paths_copy[:n_train],
        splits[1].name: paths_copy[n_train:n_train + n_val],
        splits[2].name: paths_copy[n_train + n_val:n_train + n_val + n_test],
    }


def _make_records(cfg: WaveformDatasetBuilderConfig) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Create a mapping split_name -> list of (path, class_index).
    Noise samples are generated after mechanism splits.
    """
    class_to_idx = {c: i for i, c in enumerate(cfg.classes)}
    snr_dict = None
    if cfg.snr_min is not None:
        if cfg.snr_dir is None:
            raise ValueError("snr_min requires snr_dir (directory of *_snrs*.txt files).")
        snr_dict = load_snr_values(cfg.snr_dir)

    mech_to_paths = _collect_waveform_paths(cfg.raw_root, list(cfg.mechanism_to_class.keys()))

    out: Dict[str, List[Tuple[Path, int]]] = {s.name: [] for s in cfg.splits}

    for mechanism, paths in mech_to_paths.items():
        class_label = cfg.mechanism_to_class[mechanism]
        if class_label not in class_to_idx:
            raise KeyError(f"Class label {class_label!r} for mechanism {mechanism!r} not in cfg.classes")

        split_paths = _split_paths(paths, cfg.splits, seed=stable_seed(cfg.split_seed, mechanism))
        for split_name, split_list in split_paths.items():
            cls_idx = class_to_idx[class_label]
            for path in split_list:
                if snr_dict is not None:
                    snr = extract_snr_from_filename(path, snr_dict)
                    if snr is None or snr < cfg.snr_min:
                        continue
                out[split_name].append((path, cls_idx))

    # Optional noise. Counts follow add.py when noise_counts is set (2000/600/600).
    split_names = [s.name for s in cfg.splits]
    ratios = [s.ratio for s in cfg.splits]
    counts = allocate_noise_counts(split_names, ratios, cfg.noise_total, cfg.noise_counts)
    if any(counts.values()):
        noise_idx = class_to_idx[cfg.noise_class_label]
        for split_name, n_noise in counts.items():
            out[split_name].extend((Path("__noise__"), noise_idx) for _ in range(n_noise))

    return out


def build_1d_datasets_from_raw(
    cfg: WaveformDatasetBuilderConfig,
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Returns (train_ds, val_ds, test_ds).

    Note: this is a streaming dataset (loads waveform per sample at runtime).
    """
    require_waveform_length(cfg.input_length)
    input_len = cfg.input_length

    records_by_split = _make_records(cfg)

    def make_generator(split_name: str):
        records = records_by_split[split_name]
        noise_rng = np.random.default_rng(stable_seed(cfg.split_seed, f"noise:{split_name}"))
        for path, y in records:
            if str(path) == "__noise__":
                x = noise_rng.normal(0.0, 1.0, size=(input_len,)).astype(np.float32)
            else:
                x = load_waveform_txt(path).astype(np.float32)
                require_waveform_length(x.shape[0])

                if cfg.normalize_per_sample:
                    x = (x - np.mean(x)) / (np.std(x) + 1e-8)

            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=-1)  # (L, 1)
            yield x, np.int32(y)

    output_signature = (
        tf.TensorSpec(shape=(input_len, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    train_records = records_by_split["train"]
    val_records = records_by_split["val"]
    test_records = records_by_split["test"]
    _ = (train_records, val_records, test_records)

    train_ds = tf.data.Dataset.from_generator(lambda: make_generator("train"), output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(lambda: make_generator("val"), output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(lambda: make_generator("test"), output_signature=output_signature)

    # Shuffle only train
    train_ds = train_ds.shuffle(buffer_size=min(10_000, len(records_by_split["train"])) + 1, seed=cfg.split_seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


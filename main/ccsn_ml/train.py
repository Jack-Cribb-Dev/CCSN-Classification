"""
Generic training entry points.

This will eventually:
- Read an experiment config (paths, model type, hyperparameters).
- Build datasets and models.
- Run training with callbacks and save artifacts (weights, history).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from . import datasets, models
from .data.waveform_dataset import (
    WaveformDatasetBuilderConfig,
    build_1d_datasets_from_raw,
)
from .data.waveforms import WAVEFORM_SAMPLES
from .groupings import class_scheme


@dataclass
class TrainExperimentConfig:
    base_path: Path
    experiment_name: str
    image_size: tuple = (128, 128)
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    model_type: str = "2d_a2"  # "2d_a1", "2d_a2", or "2d_grouped"
    num_classes: int = 20
    learning_rate: float = 0.005


def train_2d_experiment(cfg: TrainExperimentConfig) -> tf.keras.callbacks.History:
    """Minimal 2D training loop using the cleaned A2-style model."""
    datasets_root = cfg.base_path / "datasets" / "A"
    train_dir = datasets_root / f"Train_{cfg.experiment_name}"
    val_dir = datasets_root / f"Val_{cfg.experiment_name}"
    test_dir = datasets_root / f"Test_{cfg.experiment_name}"

    train_ds, val_ds, test_ds = datasets.load_image_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )

    model_cfg = models.ModelConfig2D(
        input_shape=(cfg.image_size[0], cfg.image_size[1], 3),
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
    )
    builders = {
        "2d_a1": models.build_2d_cnn_a1,
        "2d_a2": models.build_2d_cnn_a2,
        "2d_grouped": models.build_2d_cnn_grouped,
    }
    if cfg.model_type not in builders:
        raise ValueError(f"Unsupported model_type={cfg.model_type!r}. Expected one of {sorted(builders)}.")
    model = builders[cfg.model_type](model_cfg)

    models_dir = cfg.base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"best_model_{cfg.experiment_name}.h5"

    callbacks = [
        EarlyStopping(patience=cfg.patience, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.max_epochs,
        callbacks=callbacks,
    )

    # Evaluate on test set
    loss, acc = model.evaluate(test_ds)
    print(f"Final test accuracy ({cfg.experiment_name}): {acc:.4f}")

    # Save history for downstream plotting (mirrors FINALS).
    try:
        import pandas as pd

        history_dir = cfg.base_path / "model_results"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"history_{cfg.experiment_name}.csv"
        pd.DataFrame(history.history).to_csv(history_path, index=False)
        print(f"Saved training history to {history_path}")
    except Exception as e:
        print(f"Warning: failed to save training history CSV: {e}")

    return history


@dataclass
class Train1DExperimentConfig:
    base_path: Path  # where to write models/history
    experiment_name: str  # e.g. A1_1d

    raw_root: Path  # root with mechanism subfolders containing waveforms

    # Class scheme
    class_mode: str = "20"  # "20" or "grouped_9"
    grouping_name: str | None = None  # optional future extension

    # Dataset / training. Length is the full 4-second waveform.
    input_length: int = WAVEFORM_SAMPLES
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10

    learning_rate: float = 0.001

    # Optional noise augmentation. noise_counts matches add.py when set.
    noise_total: int = 0
    noise_counts: dict[str, int] | None = None
    noise_class_label: str = "noise"
    snr_min: float | None = None
    snr_dir: Path | None = None


def train_1d_experiment(cfg: Train1DExperimentConfig) -> tf.keras.callbacks.History:
    """
    Minimal 1D training loop built from raw `.txt` waveforms.

    Note: this is currently a clean, generic loader/trainer; for exact matching to
    your prior hyperparameter tuning runs we may need to revisit dataset splits
    and normalization settings once we compare results.
    """
    scheme = class_scheme(cfg.class_mode)
    if cfg.input_length != WAVEFORM_SAMPLES:
        raise ValueError(
            f"1D input_length must be {WAVEFORM_SAMPLES} (4 s at 4096 Hz), got {cfg.input_length}."
        )

    ds_builder_cfg = WaveformDatasetBuilderConfig(
        raw_root=cfg.raw_root,
        mechanism_to_class=scheme.mechanism_to_class,
        classes=scheme.classes,
        input_length=WAVEFORM_SAMPLES,
        normalize_per_sample=True,
        noise_total=cfg.noise_total,
        noise_counts=cfg.noise_counts,
        noise_class_label=cfg.noise_class_label,
        snr_min=cfg.snr_min,
        snr_dir=cfg.snr_dir,
    )

    train_ds, val_ds, test_ds = build_1d_datasets_from_raw(ds_builder_cfg, batch_size=cfg.batch_size)

    model_cfg = models.ModelConfig1D(
        input_length=WAVEFORM_SAMPLES,
        num_classes=scheme.num_classes,
        learning_rate=cfg.learning_rate,
    )
    if scheme.mode == "grouped_9":
        model = models.build_1d_cnn_grouped(model_cfg)
    else:
        model = models.build_1d_cnn_a1(model_cfg)

    models_dir = cfg.base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"best_model_{cfg.experiment_name}.h5"

    callbacks = [
        EarlyStopping(patience=cfg.patience, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.max_epochs,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(test_ds)
    print(f"Final 1D test accuracy ({cfg.experiment_name}): {acc:.4f}")

    # Save history for later plotting.
    try:
        import pandas as pd

        history_dir = cfg.base_path / "model_results"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"history_{cfg.experiment_name}.csv"
        pd.DataFrame(history.history).to_csv(history_path, index=False)
        print(f"Saved 1D training history to {history_path}")
    except Exception as e:
        print(f"Warning: failed to save 1D training history CSV: {e}")

    return history


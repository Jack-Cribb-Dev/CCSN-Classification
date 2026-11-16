"""
Dataset construction utilities for 1D and 2D CNNs.

This will eventually replace the ad hoc dataset loading logic scattered across
`FINALS` and earlier code, including:
- Directory-based image datasets for 2D spectrograms.
- Direct-from-waveform `tf.data` pipelines for 1D time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf


def load_image_datasets(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Generalization of your `utils.ds_loader.load_datasets`, using paths as Path objects
    and avoiding hard-coded globals.
    """

    def preprocess(ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(val_dir),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )

    return preprocess(train_ds), preprocess(val_ds), preprocess(test_ds)


# Placeholders for 1D waveform-based datasets will be added here when we
# migrate the 1D CNN pipeline.


"""
Model builders for CCSN classification.

This module will host:
- 2D CNN architectures for spectrogram inputs (including the A1/A2/C models).
- 1D CNN architectures for raw time-series inputs.
- Any future architectures (ResNet-style, transformers, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from .data.waveforms import WAVEFORM_SAMPLES
from .groupings import N_CLASSES_FULL, N_CLASSES_GROUPED


@dataclass
class ModelConfig2D:
    input_shape: Tuple[int, int, int] = (128, 128, 3)
    num_classes: int = 20
    learning_rate: float = 0.005


def build_2d_cnn_a2(cfg: ModelConfig2D) -> tf.keras.Model:
    """
    Clean version of the A2 20-class 2D CNN from FINALS/train_best_model.py.
    This keeps the architecture but removes hard-coded paths and side effects.
    """
    if cfg.num_classes != N_CLASSES_FULL:
        raise ValueError(
            f"A2 is the 20-class model ({N_CLASSES_FULL} outputs). "
            f"Grouped training must use build_2d_cnn_grouped ({N_CLASSES_GROUPED} outputs)."
        )
    model = models.Sequential()
    model.add(tf.keras.Input(shape=cfg.input_shape))

    # Conv block 0
    model.add(layers.Conv2D(128, kernel_size=4, padding="same", dilation_rate=(1, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv block 1
    model.add(layers.Conv2D(64, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv block 2
    model.add(layers.Conv2D(64, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Conv block 3
    model.add(layers.Conv2D(64, kernel_size=4, padding="same", dilation_rate=(1, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv block 4
    model.add(layers.Conv2D(128, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Conv block 5
    model.add(layers.Conv2D(32, kernel_size=3, padding="same", dilation_rate=(1, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Dense head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(384))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


@dataclass
class ModelConfig1D:
    input_length: int = WAVEFORM_SAMPLES
    num_classes: int = N_CLASSES_FULL
    learning_rate: float = 0.001


def _require_full_length(length: int) -> None:
    if length != WAVEFORM_SAMPLES:
        raise ValueError(
            f"1D models take the full waveform ({WAVEFORM_SAMPLES} samples). Got {length}."
        )


def build_1d_cnn_a1(cfg: ModelConfig1D) -> tf.keras.Model:
    """
    20-class 1D CNN from `Dir_main/100k/1D_CNN/train_model.py`.

    The original helper defaulted `input_length` to 4096. That was a 1-second
    snippet left over from an earlier experiment. The waveforms, `add.py`, and
    the grouped 1D weights all use 16384 samples, so this builder does too.
    """
    _require_full_length(cfg.input_length)
    if cfg.num_classes != N_CLASSES_FULL:
        raise ValueError(f"20-class 1D model must have {N_CLASSES_FULL} outputs, got {cfg.num_classes}.")

    model = models.Sequential()
    model.add(tf.keras.Input(shape=(cfg.input_length, 1)))

    # Conv Layer 0
    model.add(layers.Conv1D(32, kernel_size=5, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Conv Layer 1
    model.add(layers.Conv1D(64, kernel_size=5, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Conv Layer 2
    model.add(layers.Conv1D(64, kernel_size=7, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Dropout is in the old script but set to 0.0.

    # Dense head
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_1d_cnn_grouped(cfg: ModelConfig1D) -> tf.keras.Model:
    """
    9-class 1D CNN matching `best_model_C3-2.h5`.

    Three conv layers (64, 64, 64) with kernels 7, 7, 3, dropout 0.1 after the
    last conv and after the dense layer, 9-way softmax. `Hyperparam_1D.py`
    hardcoded 20 outputs even when EXPERIMENT was C3; this does not.
    """
    _require_full_length(cfg.input_length)
    if cfg.num_classes != N_CLASSES_GROUPED:
        raise ValueError(
            f"Grouped 1D model must have {N_CLASSES_GROUPED} outputs, got {cfg.num_classes}."
        )

    model = models.Sequential()
    model.add(tf.keras.Input(shape=(cfg.input_length, 1)))

    model.add(layers.Conv1D(64, kernel_size=7, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(64, kernel_size=7, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(64, kernel_size=3, padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_2d_cnn_a1(cfg: ModelConfig2D) -> tf.keras.Model:
    """
    20-class 2D CNN matching `FINALS/models/best_model_A1.h5`.

    This is the network the dissertation reports for the full 20-class task
    (learning rate 0.001). It is a different stack from `train_best_model.py`.
    """
    if cfg.num_classes != N_CLASSES_FULL:
        raise ValueError(f"20-class 2D model must have {N_CLASSES_FULL} outputs, got {cfg.num_classes}.")

    model = models.Sequential()
    model.add(tf.keras.Input(shape=cfg.input_shape))

    model.add(layers.Conv2D(32, kernel_size=3, padding="same", dilation_rate=(1, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, kernel_size=3, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, kernel_size=3, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, kernel_size=3, padding="same", dilation_rate=(1, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(320, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_2d_cnn_grouped(cfg: ModelConfig2D) -> tf.keras.Model:
    """
    9-class 2D CNN matching the C3 layer stack in `best_model_C3.h5`.

    That weight file's last dense layer was saved with 20 units. The grouped
    task has 9 labels (`gen_grouped.py` plus `add.py`), so the head here has 9.
    The saved run used Adam at 0.0005.
    """
    if cfg.num_classes != N_CLASSES_GROUPED:
        raise ValueError(
            f"Grouped 2D model must have {N_CLASSES_GROUPED} outputs, got {cfg.num_classes}."
        )

    model = models.Sequential()
    model.add(tf.keras.Input(shape=cfg.input_shape))

    model.add(layers.Conv2D(96, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(32, kernel_size=3, padding="same", dilation_rate=(1, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(96, kernel_size=3, padding="same", dilation_rate=(1, 1)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(32, kernel_size=4, padding="same", dilation_rate=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(cfg.num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


"""
Hyperparameter tuning for CCSN models.

Runs Keras Tuner (Hyperband) over 2D CNN architectures. Saves the best model
to the same path the rest of the pipeline expects, so you can run
evaluate_experiment immediately after tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

from . import datasets
from .data.waveforms import WAVEFORM_SAMPLES


def build_2d_cnn_tunable(
    hp: kt.HyperParameters,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 20,
    model_level: int = 3,
) -> tf.keras.Model:
    """
    Tunable 2D CNN for Hyperband. Based on Dir_main/100k/2D_CNN/Hyperparam.py.
    model_level 1–4 controls search space size (more layers, dilation, etc.).
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    conv_layers = model_level + 2
    filters_max = {1: 64, 2: 96, 3: 128, 4: 128}[model_level]
    filters_min = 32
    dense_min, dense_max = {1: (128, 192), 2: (192, 256), 3: (256, 320), 4: (320, 384)}[model_level]
    dropout_dense_range = {1: (0.1, 0.3), 2: (0.2, 0.4), 3: (0.3, 0.5), 4: (0.3, 0.6)}[model_level]
    dropout_conv_range = {1: (0.0, 0.2), 2: (0.1, 0.3), 3: (0.1, 0.4), 4: (0.2, 0.5)}[model_level]
    use_batchnorm = model_level >= 2
    allow_dilation = model_level >= 3

    activation_type = hp.Choice("activation", ["relu", "leaky_relu"])
    learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    dense_units = hp.Int("dense_units", dense_min, dense_max, step=32)
    dropout_dense = hp.Float("dropout_rate_dense", dropout_dense_range[0], dropout_dense_range[1], step=0.1)

    for i in range(conv_layers):
        filters = hp.Int(f"filters_{i}", filters_min, filters_max, step=32)
        kernel_size = hp.Choice(f"kernel_size_{i}", [3, 4])
        dilation = (2, 2) if (allow_dilation and hp.Boolean(f"use_dilation_{i}")) else (1, 1)
        model.add(layers.Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
        if activation_type == "leaky_relu":
            model.add(LeakyReLU())
        else:
            model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        if hp.Boolean(f"use_dropout_conv_{i}"):
            dr = hp.Float(f"dropout_rate_conv_{i}", dropout_conv_range[0], dropout_conv_range[1], step=0.1)
            model.add(layers.Dropout(dr))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_dense))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_tune_2d(
    base_path: Path,
    experiment_name: str,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    num_classes: int = 20,
    max_epochs: int = 50,
    model_level: int = 3,
    tuner_dir: str | Path | None = None,
    overwrite: bool = True,
) -> Tuple[kt.Hyperband, Dict[str, Any]]:
    """
    Load 2D datasets from base_path/datasets/A/Train_<exp>, Val_<exp>, Test_<exp>;
    run Hyperband; save best model to base_path/models/best_model_<exp>.h5
    and best hyperparameters to base_path/tuning/best_hps_<exp>.json.
    """
    datasets_root = base_path / "datasets" / "A"
    train_dir = datasets_root / f"Train_{experiment_name}"
    val_dir = datasets_root / f"Val_{experiment_name}"
    test_dir = datasets_root / f"Test_{experiment_name}"

    train_ds, val_ds, _ = datasets.load_image_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    input_shape = (image_size[0], image_size[1], 3)

    def builder(hp: kt.HyperParameters) -> tf.keras.Model:
        return build_2d_cnn_tunable(hp, input_shape=input_shape, num_classes=num_classes, model_level=model_level)

    if tuner_dir is None:
        tuner_dir = base_path / "tuning" / experiment_name
    tuner_dir = Path(tuner_dir)
    tuner_dir.mkdir(parents=True, exist_ok=True)

    tuner = kt.Hyperband(
        builder,
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=2,
        directory=str(tuner_dir),
        project_name="hyperband",
        overwrite=overwrite,
        executions_per_trial=1,
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    models_dir = base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_model_{experiment_name}.h5"
    best_model.save(best_model_path)
    print(f"Saved best 2D model to {best_model_path}")

    def _to_json_serializable(v: Any) -> Any:
        if v is None or isinstance(v, (bool, str)):
            return v
        if isinstance(v, (int, float)):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    try:
        hps_values = best_hps.values
    except AttributeError:
        hps_values = {}
    if not isinstance(hps_values, dict):
        hps_values = dict(hps_values) if hasattr(hps_values, "items") else {}
    hps_serializable = {k: _to_json_serializable(v) for k, v in hps_values.items()}
    tuning_dir = base_path / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    hps_path = tuning_dir / f"best_hps_{experiment_name}.json"
    with open(hps_path, "w") as f:
        json.dump(hps_serializable, f, indent=2)
    print(f"Saved best hyperparameters to {hps_path}")

    return tuner, hps_serializable


def run_tune_1d(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    base_path: Path,
    experiment_name: str,
    num_classes: int,
    input_length: int = WAVEFORM_SAMPLES,
    max_epochs: int = 50,
    model_level: int = 4,
    tuner_dir: str | Path | None = None,
    overwrite: bool = True,
) -> Tuple[kt.Hyperband, Dict[str, Any]]:
    """
    1D Hyperband search from `Dir_main/100k/1D_CNN/Hyperparam_1D.py`.

    The original script always finished with Dense(20), including when
    EXPERIMENT was the grouped C3 run. `num_classes` is required so a grouped
    search saves a 9-way head. Input length is the full 16384-sample waveform.
    """
    if input_length != WAVEFORM_SAMPLES:
        raise ValueError(f"1D tuning expects {WAVEFORM_SAMPLES} samples, got {input_length}.")
    if num_classes not in (9, 20):
        raise ValueError(f"num_classes must be 9 (grouped) or 20 (full). Got {num_classes}.")

    def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
        model = models.Sequential()
        model.add(layers.Input(shape=(input_length, 1)))

        conv_layers = model_level + 2
        filters_max = {1: 64, 2: 96, 3: 128, 4: 128}[model_level]
        filters_min = 32
        dense_min, dense_max = {1: (64, 128), 2: (128, 192), 3: (192, 256), 4: (256, 320)}[model_level]
        dropout_dense_range = {1: (0.1, 0.3), 2: (0.2, 0.4), 3: (0.3, 0.5), 4: (0.4, 0.6)}[model_level]
        dropout_conv_range = {1: (0.0, 0.2), 2: (0.1, 0.3), 3: (0.1, 0.4), 4: (0.2, 0.5)}[model_level]

        activation_type = hp.Choice("activation", ["relu", "leaky_relu"])
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4])
        dense_units = hp.Int("dense_units", dense_min, dense_max, step=32)
        dropout_dense = hp.Float("dropout_rate_dense", dropout_dense_range[0], dropout_dense_range[1], step=0.1)

        for i in range(conv_layers):
            filters = hp.Int(f"filters_{i}", filters_min, filters_max, step=32)
            kernel_size = hp.Choice(f"kernel_size_{i}", [3, 5, 7])
            model.add(layers.Conv1D(filters, kernel_size, padding="same"))
            if activation_type == "leaky_relu":
                model.add(LeakyReLU())
            else:
                model.add(layers.Activation("relu"))
            model.add(layers.MaxPooling1D(pool_size=2))
            if hp.Boolean(f"use_dropout_conv_{i}"):
                dr = hp.Float(f"dropout_rate_conv_{i}", dropout_conv_range[0], dropout_conv_range[1], step=0.1)
                model.add(layers.Dropout(dr))

        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(dense_units, activation="relu"))
        model.add(layers.Dropout(dropout_dense))
        model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    if tuner_dir is None:
        tuner_dir = base_path / "tuning" / f"{experiment_name}_1d"
    tuner_dir = Path(tuner_dir)
    tuner_dir.mkdir(parents=True, exist_ok=True)

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=3,
        directory=str(tuner_dir),
        project_name="hyperband",
        overwrite=overwrite,
    )
    tuner.search(train_ds, validation_data=val_ds, epochs=max_epochs, verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    models_dir = base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_model_{experiment_name}.h5"
    best_model.save(best_model_path)
    print(f"Saved best 1D model to {best_model_path}")

    hps_values = dict(best_hps.values)
    tuning_dir = base_path / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    hps_path = tuning_dir / f"best_hps_{experiment_name}.json"
    with open(hps_path, "w", encoding="utf-8") as f:
        json.dump({k: (v if isinstance(v, (bool, int, float, str)) or v is None else str(v)) for k, v in hps_values.items()}, f, indent=2)
    print(f"Saved best 1D hyperparameters to {hps_path}")
    return tuner, hps_values

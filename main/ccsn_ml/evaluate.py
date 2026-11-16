"""
Evaluation utilities for trained CCSN classifiers.

This will absorb and clean up logic from:
- FINALS/generate_plots.py
- FINALS/utils/eval_metrics.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE

from . import datasets
from .groupings import class_scheme
from .data.waveform_dataset import WaveformDatasetBuilderConfig, build_1d_datasets_from_raw
from .data.waveforms import WAVEFORM_SAMPLES


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: Iterable[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """Generic confusion matrix plotting with proper label handling."""
    labels = list(class_labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def evaluate_2d_experiment(
    base_path: Path,
    experiment_name: str,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    class_labels: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a trained 2D model and compute predictions + confusion matrix
    on its test set. Returns (y_true, y_pred).
    """
    model_path = base_path / "models" / f"best_model_{experiment_name}.h5"
    datasets_root = base_path / "datasets" / "A"
    train_dir = datasets_root / f"Train_{experiment_name}"
    val_dir = datasets_root / f"Val_{experiment_name}"
    test_dir = datasets_root / f"Test_{experiment_name}"

    model = tf.keras.models.load_model(model_path)
    _, _, test_ds = datasets.load_image_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    if class_labels is None:
        # `image_dataset_from_directory` orders classes lexicographically by folder name.
        # Replicate that so the confusion matrix axes match label indices.
        class_labels = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    results_dir = base_path / "results" / experiment_name
    plot_confusion_matrix(
        cm,
        class_labels=class_labels,
        normalize=True,
        title=f"Confusion Matrix ({experiment_name})",
        save_path=results_dir / f"confusion_matrix_{experiment_name}.png",
    )

    # Plot training curves if history exists.
    try:
        import pandas as pd

        history_path = base_path / "model_results" / f"history_{experiment_name}.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            results_dir.mkdir(parents=True, exist_ok=True)

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            axs[0].plot(history["accuracy"], label="Train Acc", color="navy")
            axs[0].plot(history["val_accuracy"], label="Val Acc", color="darkred")
            axs[0].set_title("Training vs Validation Accuracy")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Accuracy")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(history["loss"], label="Train Loss", color="navy")
            axs[1].plot(history["val_loss"], label="Val Loss", color="darkred")
            axs[1].set_title("Training vs Validation Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss")
            axs[1].legend()
            axs[1].grid(True)

            plt.suptitle("Training Curves", fontsize=14)
            plt.tight_layout()
            fig_path = results_dir / f"training_curves_{experiment_name}.png"
            plt.savefig(fig_path)
            plt.close()
    except Exception as e:
        print(f"Warning: failed to plot training curves: {e}")

    return y_true, y_pred


def evaluate_1d_experiment(
    base_path: Path,
    experiment_name: str,
    raw_root: Path,
    class_mode: str = "20",
    input_length: int = WAVEFORM_SAMPLES,
    batch_size: int = 32,
    class_labels: List[str] | None = None,
    noise_total: int = 0,
    noise_counts: dict[str, int] | None = None,
    noise_class_label: str = "noise",
    snr_min: float | None = None,
    snr_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a trained 1D model and evaluate on the test set (rebuilt from raw_root).
    Returns (y_true, y_pred). Saves confusion matrix and training curves if history exists.
    Pass noise_total/noise_class_label if training used synthetic noise so the test set matches.
    """
    scheme = class_scheme(class_mode)
    if input_length != WAVEFORM_SAMPLES:
        raise ValueError(
            f"1D evaluation expects {WAVEFORM_SAMPLES} samples, got {input_length}."
        )

    if class_labels is None:
        class_labels = scheme.classes

    ds_cfg = WaveformDatasetBuilderConfig(
        raw_root=raw_root,
        mechanism_to_class=scheme.mechanism_to_class,
        classes=scheme.classes,
        input_length=WAVEFORM_SAMPLES,
        normalize_per_sample=True,
        noise_total=noise_total,
        noise_counts=noise_counts,
        noise_class_label=noise_class_label,
        snr_min=snr_min,
        snr_dir=snr_dir,
    )
    _, _, test_ds = build_1d_datasets_from_raw(ds_cfg, batch_size=batch_size)

    model_path = base_path / "models" / f"best_model_{experiment_name}.h5"
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred, prob_chunks = [], [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        prob_chunks.append(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.vstack(prob_chunks) if prob_chunks else np.empty((0, scheme.num_classes))
    cm = confusion_matrix(y_true, y_pred)

    results_dir = base_path / "results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        cm,
        class_labels=class_labels,
        normalize=True,
        title=f"Confusion Matrix 1D ({experiment_name})",
        save_path=results_dir / f"confusion_matrix_1d_{experiment_name}.png",
    )

    # Training curves if history exists
    try:
        import pandas as pd

        history_path = base_path / "model_results" / f"history_{experiment_name}.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].plot(history["accuracy"], label="Train Acc", color="navy")
            axs[0].plot(history["val_accuracy"], label="Val Acc", color="darkred")
            axs[0].set_title("Training vs Validation Accuracy")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Accuracy")
            axs[0].legend()
            axs[0].grid(True)
            axs[1].plot(history["loss"], label="Train Loss", color="navy")
            axs[1].plot(history["val_loss"], label="Val Loss", color="darkred")
            axs[1].set_title("Training vs Validation Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss")
            axs[1].legend()
            axs[1].grid(True)
            plt.suptitle("Training Curves (1D)", fontsize=14)
            plt.tight_layout()
            plt.savefig(results_dir / f"training_curves_1d_{experiment_name}.png")
            plt.close()
    except Exception as e:
        print(f"Warning: failed to plot 1D training curves: {e}")

    # Confidence histogram from 1D_CNN/1d_gen.py.
    if len(y_true):
        confidences = np.max(probs, axis=1)
        correct = y_pred == y_true
        plt.figure(figsize=(10, 5))
        plt.hist(confidences[correct], bins=20, alpha=0.7, label="Correct")
        plt.hist(confidences[~correct], bins=20, alpha=0.7, label="Incorrect")
        plt.xlabel("Model confidence")
        plt.ylabel("Number of samples")
        plt.title(f"Confidence distribution ({experiment_name})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f"confidence_histogram_1d_{experiment_name}.png")
        plt.close()

    # t-SNE of the penultimate layer, as in 1D_CNN/eval_1d.py.
    # Rebuild the test set: the first pass may have been cached, but a fresh
    # iterator keeps this independent of how many batches predict() consumed.
    try:
        feature_layer = model.layers[-3].output
        feature_model = tf.keras.Model(inputs=model.input, outputs=feature_layer)
        _, _, test_ds_features = build_1d_datasets_from_raw(ds_cfg, batch_size=batch_size)
        feature_chunks = []
        for x_batch, _ in test_ds_features:
            feature_chunks.append(feature_model.predict(x_batch, verbose=0))
        if feature_chunks:
            features = np.vstack(feature_chunks)
            perplexity = min(30, max(5, len(features) - 1))
            embedding = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            ).fit_transform(features)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap="tab20", alpha=0.7, s=10)
            cbar = plt.colorbar(scatter)
            cbar.set_ticks(np.arange(len(class_labels)))
            cbar.set_ticklabels(list(class_labels))
            plt.title(f"t-SNE projection 1D ({experiment_name})")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(results_dir / f"tsne_projection_1d_{experiment_name}.png")
            plt.close()
    except Exception as e:
        print(f"Warning: failed to plot 1D t-SNE: {e}")

    return y_true, y_pred


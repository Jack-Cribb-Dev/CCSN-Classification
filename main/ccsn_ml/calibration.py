"""
Calibration analysis: reliability diagrams, ECE, and confidence-based error stats.

This consolidates logic from FINALS/error_analysis.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from . import datasets


def reliability_diagram(
    y_true: np.ndarray,
    confidences: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Path | None = None,
) -> float:
    """Plot reliability diagram and return Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    accuracies = np.zeros(num_bins)
    confidences_avg = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            accuracies[i] = np.mean(y_true[in_bin] == y_pred[in_bin])
            confidences_avg[i] = np.mean(confidences[in_bin])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.plot(confidences_avg, accuracies, marker="o", color="navy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()

    ece = np.sum(np.abs(confidences_avg - accuracies) * bin_counts) / np.sum(bin_counts)
    return float(ece)


def analyze_model_calibration(
    base_path: Path,
    experiment_name: str,
    image_size: tuple = (128, 128),
    batch_size: int = 32,
) -> None:
    """Run calibration analysis on a 2D model (reliability diagram, basic stats)."""
    model_path = base_path / "models" / f"best_model_{experiment_name}.h5"
    datasets_root = base_path / "datasets" / "A"
    train_dir = datasets_root / f"Train_{experiment_name}"
    val_dir = datasets_root / f"Val_{experiment_name}"
    test_dir = datasets_root / f"Test_{experiment_name}"

    model = tf.keras.models.load_model(model_path)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    y_true, y_pred, probs_all = [], [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        probs_all.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs_all = np.vstack(probs_all)
    confidences = np.max(probs_all, axis=1)

    results_dir = base_path / "results" / experiment_name
    ece = reliability_diagram(
        y_true,
        confidences,
        y_pred,
        num_bins=10,
        title=f"Reliability Diagram ({experiment_name})",
        save_path=results_dir / f"reliability_diagram_{experiment_name}.png",
    )
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    high_conf_wrong = (confidences > 0.9) & (y_true != y_pred)
    low_conf_correct = (confidences < 0.6) & (y_true == y_pred)
    print(f"High-confidence errors (>0.9): {np.sum(high_conf_wrong)}")
    print(f"Low-confidence correct (<0.6): {np.sum(low_conf_correct)}")


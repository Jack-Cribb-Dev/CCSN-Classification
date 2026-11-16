"""
SNR-stratified evaluation plots migrated from `FINALS/generate_plots.py`.

These plots require:
- A trained 2D model.
- A test image directory produced by `image_dataset_from_directory`.
- SNR log files (`snrs_logs/*`) that can be loaded by `ccsn_ml.snr.load_snr_values`.

The original FINALS pipeline assumed a filename->SNR mapping via:
- PNG filename stem: `<model>_<index>`
- `snr_dict[model][index]`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with FINALS; some users may extend)
import tensorflow as tf
from sklearn.metrics import classification_report

from .snr import extract_snr_from_filename, load_snr_values
from .datasets import load_image_datasets


def plot_snr_accuracy_curves(
    *,
    base_path: Path,
    experiment_name: str,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 32,
    snr_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    bins: np.ndarray | None = None,
) -> None:
    """
    Reproduce the FINALS plots:
    - accuracy vs SNR (binned)
    - per-class accuracy vs SNR
    - correct vs incorrect SNR histograms (raw + normalized)
    """
    if bins is None:
        bins = np.arange(20, 55, 5)

    datasets_root = base_path / "datasets" / "A"
    if test_dir is None:
        test_dir = datasets_root / f"Test_{experiment_name}"
    if results_dir is None:
        results_dir = base_path / "results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if snr_dir is None:
        snr_dir = base_path / "snrs_logs"

    model_path = base_path / "models" / f"best_model_{experiment_name}.h5"
    model = tf.keras.models.load_model(model_path)

    # Load datasets (same structure as FINALS)
    train_dir = datasets_root / f"Train_{experiment_name}"
    val_dir = datasets_root / f"Val_{experiment_name}"
    _, _, test_ds = load_image_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Predictions
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filepaths used to match SNR indices (FINALS relied on ordering alignment)
    filepaths = sorted(glob.glob(os.path.join(str(test_dir), "**", "*.png"), recursive=True))
    if len(filepaths) != len(y_true):
        # Keep behavior similar to FINALS: trim to smallest length
        n = min(len(filepaths), len(y_true))
        filepaths = filepaths[:n]
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    snr_dict = load_snr_values(snr_dir)

    snr_values, correctness = [], []
    for i, path in enumerate(filepaths):
        snr = extract_snr_from_filename(path, snr_dict)
        snr_values.append(snr)
        correctness.append(int(y_true[i] == y_pred[i]))

    snr_values = np.array(snr_values, dtype=object)
    correctness = np.array(correctness, dtype=int)

    valid_mask = np.array([v is not None for v in snr_values])
    snr_values = snr_values[valid_mask].astype(float)
    correctness = correctness[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Accuracy vs SNR
    bin_centers = bins[:-1] + 2.5
    acc_per_bin = []
    for i in range(len(bins) - 1):
        mask = (snr_values >= bins[i]) & (snr_values < bins[i + 1])
        acc_per_bin.append(np.mean(correctness[mask]) if np.sum(mask) > 0 else np.nan)

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, acc_per_bin, marker="o", color="dodgerblue")
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. SNR ({experiment_name})")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig(results_dir / f"snr_accuracy_{experiment_name}.png")
    plt.close()

    # Per-class accuracy vs SNR
    class_names = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Exclude noise-like class names from per-class SNR curves (matches FINALS intent)
    is_noise = lambda s: str(s).lower() == "noise"
    per_class_acc: Dict[int, List[float]] = {i: [] for i in range(len(class_names)) if not is_noise(class_names[i])}

    for cls_idx in per_class_acc.keys():
        cls_mask = y_true_valid == cls_idx
        for i in range(len(bins) - 1):
            bin_mask = (snr_values >= bins[i]) & (snr_values < bins[i + 1])
            combined = cls_mask & bin_mask
            if np.sum(combined) > 0:
                per_class_acc[cls_idx].append(float(np.mean(correctness[combined])))
            else:
                per_class_acc[cls_idx].append(np.nan)

    plt.figure(figsize=(10, 6))
    for cls_idx, accs in per_class_acc.items():
        plt.plot(bin_centers, accs, marker="o", label=idx_to_class[cls_idx])
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy vs. SNR ({experiment_name})")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(results_dir / f"per_class_snr_accuracy_{experiment_name}.png")
    plt.close()

    # SNR histogram: correct vs incorrect
    fine_bins = np.arange(20, 51, 1)
    correct_mask = correctness == 1
    incorrect_mask = correctness == 0
    correct_hist, _ = np.histogram(snr_values[correct_mask], bins=fine_bins)
    incorrect_hist, _ = np.histogram(snr_values[incorrect_mask], bins=fine_bins)

    bin_centers_fine = fine_bins[:-1] + 1
    bar_width = 1.8

    plt.figure(figsize=(8, 5))
    plt.bar(
        bin_centers_fine,
        correct_hist,
        width=bar_width,
        color="#1c1c94",
        label="Classified correctly",
    )
    plt.bar(
        bin_centers_fine,
        incorrect_hist,
        width=bar_width,
        bottom=correct_hist,
        color="#800000",
        label="Misclassified",
    )
    plt.xlabel("SNR")
    plt.ylabel("Number of signals")
    plt.title(f"SNR Histogram: Correct vs Incorrect ({experiment_name})")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / f"snr_histogram_{experiment_name}.png")
    plt.close()

    # Normalized SNR histogram (percentage in each bin)
    bins_norm = np.arange(20, 55, 2.5)
    bin_centers_norm = (bins_norm[:-1] + bins_norm[1:]) / 2
    correct_counts, incorrect_counts = [], []
    for i in range(len(bins_norm) - 1):
        mask = (snr_values >= bins_norm[i]) & (snr_values < bins_norm[i + 1])
        correct_in_bin = int(np.sum(correctness[mask] == 1))
        incorrect_in_bin = int(np.sum(correctness[mask] == 0))
        total = correct_in_bin + incorrect_in_bin
        if total > 0:
            correct_counts.append(correct_in_bin / total * 100.0)
            incorrect_counts.append(incorrect_in_bin / total * 100.0)
        else:
            correct_counts.append(np.nan)
            incorrect_counts.append(np.nan)

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers_norm, incorrect_counts, width=2.5, color="#8B0000", label="Incorrect", alpha=0.8)
    plt.bar(bin_centers_norm, correct_counts, width=2.5, bottom=incorrect_counts, color="#1E3F66", label="Correct", alpha=0.8)
    plt.xlabel("SNR")
    plt.ylabel("Percentage (%)")
    plt.title(f"SNR Classification Success ({experiment_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(results_dir / f"snr_histogram_normalized_{experiment_name}.png")
    plt.close()

    # Classification report (optional but handy)
    try:
        # Use sorted class names for report indices
        print(classification_report(y_true_valid, y_pred_valid, target_names=class_names, digits=4))
    except Exception:
        pass


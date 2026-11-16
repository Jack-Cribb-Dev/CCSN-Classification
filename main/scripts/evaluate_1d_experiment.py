"""Evaluate a trained 1D CCSN model (confusion matrix + training curves)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ccsn_ml.data.waveforms import WAVEFORM_SAMPLES
from ccsn_ml.evaluate import evaluate_1d_experiment
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained 1D CCSN model.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. exp_1d_A1_20class.yaml).")
    args = parser.parse_args()

    config = load_config(args.config)
    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]
    raw_root = Path(config["raw_root"])
    class_mode = config.get("class_mode", "20")
    input_length = int(config["model"].get("input_length", WAVEFORM_SAMPLES))
    batch_size = int(config["dataset"]["batch_size"])
    noise_total = int(config.get("noise_total", 0))
    noise_counts = config.get("noise_counts")
    noise_class_label = config.get("noise_class_label", "noise")
    snr_min = config.get("snr_min")
    snr_dir = Path(config["snr_dir"]) if config.get("snr_dir") else None

    y_true, y_pred = evaluate_1d_experiment(
        base_path=base_path,
        experiment_name=experiment_name,
        raw_root=raw_root,
        class_mode=class_mode,
        input_length=input_length,
        batch_size=batch_size,
        noise_total=noise_total,
        noise_counts=noise_counts,
        noise_class_label=noise_class_label,
        snr_min=snr_min,
        snr_dir=snr_dir,
    )
    acc = float(np.mean(y_true == y_pred))
    print(f"1D test accuracy ({experiment_name}): {acc:.4f}")


if __name__ == "__main__":
    main()

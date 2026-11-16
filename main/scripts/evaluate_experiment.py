from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.evaluate import evaluate_2d_experiment
from ccsn_ml.snr_analysis import plot_snr_accuracy_curves
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CCSN model experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)

    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]

    image_size = tuple(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])

    y_true, y_pred = evaluate_2d_experiment(
        base_path=base_path,
        experiment_name=experiment_name,
        image_size=image_size,
        batch_size=batch_size,
    )
    # Printing summary to help quick CLI usage.
    import numpy as np

    acc = float(np.mean(y_true == y_pred))
    print(f"Accuracy ({experiment_name}): {acc:.4f}")

    # If SNR logs exist, also generate the SNR-stratified plots.
    snr_dir = base_path / "snrs_logs"
    if snr_dir.exists():
        try:
            plot_snr_accuracy_curves(
                base_path=base_path,
                experiment_name=experiment_name,
                image_size=image_size,
                batch_size=batch_size,
                snr_dir=snr_dir,
            )
        except Exception as e:
            print(f"Warning: failed to generate SNR plots: {e}")


if __name__ == "__main__":
    main()


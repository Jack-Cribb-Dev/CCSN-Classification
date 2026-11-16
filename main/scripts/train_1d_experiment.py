from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.data.waveforms import WAVEFORM_SAMPLES
from ccsn_ml.train import Train1DExperimentConfig, train_1d_experiment
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 1D CCSN model from raw waveform .txt files.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)

    cfg = Train1DExperimentConfig(
        base_path=Path(config["base_path"]),
        experiment_name=config["experiment_name"],
        raw_root=Path(config["raw_root"]),
        class_mode=config.get("class_mode", "20"),
        input_length=int(config["model"].get("input_length", WAVEFORM_SAMPLES)),
        batch_size=int(config["dataset"]["batch_size"]),
        max_epochs=int(config["training"]["max_epochs"]),
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["model"]["learning_rate"]),
        noise_total=int(config.get("noise_total", 0)),
        noise_counts=config.get("noise_counts"),
        noise_class_label=config.get("noise_class_label", "noise"),
        snr_min=config.get("snr_min"),
        snr_dir=Path(config["snr_dir"]) if config.get("snr_dir") else None,
    )

    train_1d_experiment(cfg)


if __name__ == "__main__":
    main()


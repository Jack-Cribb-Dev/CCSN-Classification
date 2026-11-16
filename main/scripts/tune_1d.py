"""Run Hyperband tuning for the 1D CNN. Grouped runs use a 9-way head."""
from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.data.waveform_dataset import WaveformDatasetBuilderConfig, build_1d_datasets_from_raw
from ccsn_ml.data.waveforms import WAVEFORM_SAMPLES
from ccsn_ml.groupings import class_scheme
from ccsn_ml.tuning import run_tune_1d
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the 1D CCSN model.")
    parser.add_argument("--config", required=True, help="Path to a 1D YAML config.")
    parser.add_argument("--model-level", type=int, default=4, help="Tuner search depth (1-4). Default 4.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max epochs per trial.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite an existing tuner run.")
    args = parser.parse_args()

    config = load_config(args.config)
    scheme = class_scheme(config.get("class_mode", "20"))
    input_length = int(config["model"].get("input_length", WAVEFORM_SAMPLES))
    batch_size = int(config["dataset"]["batch_size"])
    max_epochs = int(config["training"].get("max_epochs", 50) if args.max_epochs is None else args.max_epochs)

    ds_cfg = WaveformDatasetBuilderConfig(
        raw_root=Path(config["raw_root"]),
        mechanism_to_class=scheme.mechanism_to_class,
        classes=scheme.classes,
        input_length=input_length,
        normalize_per_sample=True,
        noise_total=int(config.get("noise_total", 0)),
        noise_counts=config.get("noise_counts"),
        noise_class_label=config.get("noise_class_label", "noise"),
        snr_min=config.get("snr_min"),
        snr_dir=Path(config["snr_dir"]) if config.get("snr_dir") else None,
    )
    train_ds, val_ds, _ = build_1d_datasets_from_raw(ds_cfg, batch_size=batch_size)

    run_tune_1d(
        train_ds,
        val_ds,
        base_path=Path(config["base_path"]),
        experiment_name=config["experiment_name"],
        num_classes=scheme.num_classes,
        input_length=input_length,
        max_epochs=max_epochs,
        model_level=args.model_level,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()

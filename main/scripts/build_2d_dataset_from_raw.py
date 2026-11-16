from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.data.spectrogram_dataset import (
    SpectrogramDatasetBuilderConfig,
    DatasetSplit,
)
from ccsn_ml.data.preprocess import SpectrogramConfig
from ccsn_ml.groupings import class_scheme
from ._config import load_config
from ccsn_ml.data.spectrogram_dataset import build_2d_spectrogram_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2D spectrogram dataset from raw waveform .txt files.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_root = Path(config["raw_root"])
    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]
    dataset_variant = config.get("dataset_variant", "A")
    class_mode = config.get("class_mode", "20")  # "20" or "grouped_9"

    noise_total = int(config.get("noise_total", 0))
    noise_counts = config.get("noise_counts")
    noise_class_label = config.get("noise_class_label", None)
    scheme = class_scheme(class_mode)

    spectrogram_cfg = SpectrogramConfig()
    spectrogram_cfg_dict = config.get("spectrogram", None)
    if spectrogram_cfg_dict:
        # Override only known fields
        for k, v in spectrogram_cfg_dict.items():
            if hasattr(spectrogram_cfg, k):
                setattr(spectrogram_cfg, k, v)

    builder_cfg = SpectrogramDatasetBuilderConfig(
        raw_root=raw_root,
        output_base=base_path / "datasets",
        dataset_variant=dataset_variant,
        experiment_name=experiment_name,
        mechanism_to_class=scheme.mechanism_to_class,
        classes=list(scheme.classes),
        split_seed=int(config.get("split_seed", 42)),
        spectrogram_cfg=spectrogram_cfg,
        noise_total=noise_total,
        noise_counts=noise_counts,
        noise_class_label=(noise_class_label if noise_class_label is not None else scheme.mechanism_to_class.get("noise", "noise")),
        snr_min=config.get("snr_min"),
        snr_dir=Path(config["snr_dir"]) if config.get("snr_dir") else None,
        splits=(
            DatasetSplit("train", float(config.get("train_ratio", 0.7))),
            DatasetSplit("val", float(config.get("val_ratio", 0.15))),
            DatasetSplit("test", float(config.get("test_ratio", 0.15))),
        ),
    )

    build_2d_spectrogram_dataset(builder_cfg)


if __name__ == "__main__":
    main()


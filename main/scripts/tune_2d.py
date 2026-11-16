"""Run Hyperband tuning for the 2D CNN; save best model and hyperparameters."""
from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.groupings import class_scheme
from ccsn_ml.tuning import run_tune_2d
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for 2D CCSN model.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. exp_A2_20class_2d.yaml).")
    parser.add_argument("--model-level", type=int, default=3, help="Tuner search depth (1-4). Default 3.")
    parser.add_argument("--max-epochs", type=int, default=50, help="Max epochs per trial.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing tuner run.")
    args = parser.parse_args()

    config = load_config(args.config)
    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]
    image_size = tuple(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])
    scheme = class_scheme(config.get("class_mode", "20"))
    yaml_classes = int(config["model"]["num_classes"])
    if yaml_classes != scheme.num_classes:
        raise SystemExit(
            f"model.num_classes is {yaml_classes}, but class_mode {scheme.mode!r} "
            f"has {scheme.num_classes} outputs."
        )
    num_classes = scheme.num_classes
    max_epochs = int(config["training"].get("max_epochs", args.max_epochs))
    if args.max_epochs != 50:
        max_epochs = args.max_epochs

    run_tune_2d(
        base_path=base_path,
        experiment_name=experiment_name,
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        max_epochs=max_epochs,
        model_level=args.model_level,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()

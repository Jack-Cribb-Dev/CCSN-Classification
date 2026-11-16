from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.groupings import class_scheme
from ccsn_ml.train import TrainExperimentConfig, train_2d_experiment
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CCSN model experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)

    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]

    # Dataset
    image_size = tuple(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])

    model_type = config["model"]["type"]
    scheme = class_scheme(config.get("class_mode", "20"))
    yaml_classes = int(config["model"]["num_classes"])
    if yaml_classes != scheme.num_classes:
        raise SystemExit(
            f"model.num_classes is {yaml_classes}, but class_mode {scheme.mode!r} "
            f"has {scheme.num_classes} outputs."
        )
    if scheme.mode == "grouped_9" and model_type != "2d_grouped":
        raise SystemExit("Grouped 2D training requires model.type: 2d_grouped (9 outputs).")
    if scheme.mode == "20" and model_type == "2d_grouped":
        raise SystemExit("model.type 2d_grouped is the 9-class network. Set class_mode: grouped_9.")

    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["patience"])
    learning_rate = float(config["model"].get("learning_rate", 0.005))

    cfg = TrainExperimentConfig(
        base_path=base_path,
        experiment_name=experiment_name,
        image_size=image_size,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        model_type=model_type,
        num_classes=scheme.num_classes,
        learning_rate=learning_rate,
    )

    train_2d_experiment(cfg)


if __name__ == "__main__":
    main()


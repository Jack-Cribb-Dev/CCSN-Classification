from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.calibration import analyze_model_calibration
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration analysis for a trained model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]

    image_size = tuple(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])

    analyze_model_calibration(
        base_path=base_path,
        experiment_name=experiment_name,
        image_size=image_size,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()


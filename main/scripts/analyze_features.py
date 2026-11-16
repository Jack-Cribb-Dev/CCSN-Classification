from __future__ import annotations

import argparse
from pathlib import Path

from ccsn_ml.features import plot_tsne_and_pca
from ccsn_ml.groupings import class_scheme
from ._config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-space analysis for a trained model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    base_path = Path(config["base_path"])
    experiment_name = config["experiment_name"]

    # image_dataset_from_directory labels folders alphabetically, which matches
    # the 20-class list but not the 1D grouped index order.
    test_dir = base_path / "datasets" / "A" / f"Test_{experiment_name}"
    if test_dir.is_dir():
        labels = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    else:
        labels = class_scheme(config.get("class_mode", "20")).classes
    plot_tsne_and_pca(base_path=base_path, experiment_name=experiment_name, class_labels=labels)


if __name__ == "__main__":
    main()


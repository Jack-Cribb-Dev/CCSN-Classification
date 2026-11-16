"""
Feature-space analysis (t-SNE, PCA) for CCSN models.

This will encapsulate the logic from FINALS/analyse_features.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import datasets
from .groupings import MECHANISMS_20


def extract_embeddings(
    base_path: Path,
    experiment_name: str,
    layer_index_from_end: int = 3,
    image_size: tuple = (128, 128),
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) for the penultimate layer of a 2D model."""
    model_path = base_path / "models" / f"best_model_{experiment_name}.h5"
    datasets_root = base_path / "datasets" / "A"
    train_dir = datasets_root / f"Train_{experiment_name}"
    val_dir = datasets_root / f"Val_{experiment_name}"
    test_dir = datasets_root / f"Test_{experiment_name}"

    model = tf.keras.models.load_model(model_path)
    _, _, test_ds = datasets.load_image_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    intermediate_layer = model.layers[-layer_index_from_end].output
    feature_model = tf.keras.Model(inputs=model.input, outputs=intermediate_layer)

    features, labels = [], []
    for images, batch_labels in test_ds:
        feats = feature_model.predict(images, verbose=0)
        features.append(feats)
        labels.extend(batch_labels.numpy())

    return np.vstack(features), np.array(labels)


def plot_tsne_and_pca(
    base_path: Path,
    experiment_name: str,
    class_labels: list[str] | None = None,
) -> None:
    if class_labels is None:
        class_labels = MECHANISMS_20

    features, labels = extract_embeddings(base_path, experiment_name)
    num_classes = len(class_labels)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    emb_tsne = tsne.fit_transform(features)

    print("Running PCA...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(features)

    results_dir = base_path / "results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.get_cmap("hsv", num_classes)

    # t-SNE
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels, cmap=cmap, alpha=0.7, s=10)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(class_labels)
    cbar.ax.tick_params(labelsize=8)
    plt.title(f"t-SNE Projection ({experiment_name})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / f"tsne_projection_{experiment_name}.png")
    plt.close()

    # PCA
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels, cmap=cmap, alpha=0.7, s=10)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(class_labels)
    cbar.ax.tick_params(labelsize=8)
    plt.title(f"PCA Projection ({experiment_name})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / f"pca_projection_{experiment_name}.png")
    plt.close()


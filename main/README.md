---
id: 20260318T184702-ccsn-classification-clean-pipeline
type: research
sensitivity: L0
created: '2026-03-18T18:47:02.736663+00:00'
tags: []
---

## CCSN Classification – Clean Pipeline

This folder contains a cleaned, modular implementation of your CCSN classification pipeline, intended for:

- Running on any machine (laptop, workstation, or cluster).
- Supporting both 1D (time-series) and 2D (spectrogram) CNN models.
- Exploring multiple class groupings and SNR ranges.

### Structure

- `ccsn_ml/` – Python package with reusable code:
  - `data/` – loading raw waveform `.txt` files, preprocessing, spectrogram generation.
  - `datasets.py` – building `tf.data` datasets for 1D and 2D inputs.
  - `models.py` – model builders for 1D/2D CNNs (and future architectures).
  - `train.py` – generic training entry points.
  - `evaluate.py` – evaluation utilities (confusion matrices, SNR curves, etc.).
  - `features.py` – feature-space analysis (t-SNE, PCA).
  - `calibration.py` – reliability diagrams and calibration metrics.
  - `groupings.py` – class labels and different grouping schemes.
- `configs/` – experiment configuration files (YAML).
- `scripts/` – CLI entry points for training, evaluation, and dataset building.

### Usage (high level)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run a training experiment (example config):
   ```bash
   cd CCSN_Classification/main
   python -m scripts.train_experiment --config configs/exp_A2_20class_2d.yaml
   ```
3. Run evaluation/plotting for a trained model:
   ```bash
   python -m scripts.evaluate_experiment --config configs/exp_A2_20class_2d.yaml
   ```

4. (Optional) Build spectrogram datasets from raw waveforms:
   ```bash
   python -m scripts.build_2d_dataset_from_raw --config configs/build_dataset_2d_from_raw_20class.yaml
   ```
5. **1D pipeline** (from raw waveform `.txt` folders):
   ```bash
   python -m scripts.train_1d_experiment --config configs/exp_1d_A1_20class.yaml
   python -m scripts.evaluate_1d_experiment --config configs/exp_1d_A1_20class.yaml
   ```

Set `BASE_PATH` and (for 1D / dataset building) `RAW_ROOT` via environment or by editing the YAML. See configs for examples.

For the **full workflow** with new data (raw → sort → SNRs → spectrograms → tuning → train → results), see **[PIPELINE.md](PIPELINE.md)**.

### Important notes from your project

- SNR values exist per waveform sample via log files; the pipeline must keep the mapping consistent.
- Spectrograms are generated using a specific recipe (paper-consistent STFT parameters + log scaling + histogram equalization + resizing). We will keep that recipe centralized and configurable.
---
id: 20260318T220021-full-pipeline-new-data-results
type: research
sensitivity: L0
created: '2026-03-18T22:00:21.058744+00:00'
tags: []
---

# Full pipeline: new data → results

When you get new data, run the steps below in order. The codebase supports all of them; some steps are scripts, others are one-time organisation or config.

---

## 1. Take raw data

- You receive raw waveform files (e.g. one `.txt` per waveform, one column, ~16k samples at 4096 Hz).
- Naming convention: `{papername+year}_{ccsn_ID}.txt` (e.g. `mezzacappa2020_345.txt`).

---

## 2. Sort into mechanisms

- **Goal:** One folder per mechanism (class), each containing the `.txt` files for that mechanism.
- **Target layout:**  
  `RAW_ROOT/<mechanism>/<file>.txt`  
  e.g. `RAW_ROOT/C15-3D/mezzacappa2020_345.txt`, `RAW_ROOT/NR/pan2020_0123.txt`.
- **How:** Depends how the new data is delivered. If it’s one folder per paper, you’ll need to map (paper + run ID) → mechanism (e.g. from a metadata table or naming convention), then copy/move files into the right mechanism folders. A small script or notebook that reads a manifest and does the copy is enough; the rest of the pipeline only assumes this folder structure exists.

---

## 3. Match to SNRs

- **Goal:** For each waveform file, you have a single SNR value. The rest of the pipeline expects **one aggregate SNR file per paper**: `{papername+year}_snrs.txt` (or `*_snrs-2.txt`), with **one float per line**, in the **same order** as the log rows (so line index = file index when filenames are `paperyear_0.txt`, `paperyear_1.txt`, …).
- **If you have per-paper logs** (one row per waveform, with mechanism, phase, amp, SNR, etc.): use  
  `python -m scripts.build_snr_aggregate_from_log --log path/to/log_mezzacappa2020.txt --output BASE_PATH/snrs_logs/mezzacappa2020_snrs.txt --snr-column N`  
  where `N` is the 0-based column index of the SNR in each line. Run once per paper; keep row order the same as when you list/sort the `.txt` files for that paper.
- **Canonical location:** Put all `*_snrs*.txt` files in `BASE_PATH/snrs_logs/`. The evaluation and SNR-vs-accuracy scripts read from there.

---

## 4. Convert to spectrograms (2D model)

- **Goal:** Turn each waveform into a spectrogram image so the 2D CNN can be trained.
- **Script:**  
  `python -m scripts.build_2d_dataset_from_raw --config configs/build_dataset_2d_from_raw_20class.yaml`
- **Config:** Set `base_path`, `raw_root`, `experiment_name`, `class_mode` (e.g. `"20"` or `"grouped_9"`), split ratios, and optionally `noise_total` if you’re adding a synthetic noise class.
- **Output:**  
  `BASE_PATH/datasets/A/Train_<experiment_name>/<class>/*.png`, and same for `Val_` and `Test_`.  
  Spectrogram recipe is fixed (paper-consistent STFT + log + histogram equalisation + resize) in `ccsn_ml/data/preprocess.py`.
- **Naming:** Output PNGs keep the same stem as the input `.txt` (e.g. `mezzacappa2020_345.png`) so SNR lookup by `(paper, index)` still works.

---

## 5. Hyperparameter tuning (2D and 1D)

- **Goal:** Find a good architecture and training setup for this dataset; “best” will change when data or task changes.
- **2D:**  
  `python -m scripts.tune_2d --config configs/exp_A2_20class_2d.yaml`  
  Uses the same `base_path` and dataset dirs as training. Runs Keras Tuner (Hyperband), then saves the **best model** to `BASE_PATH/models/best_model_<experiment_name>.h5` and optionally the best hyperparameters to `BASE_PATH/tuning/best_hps_<experiment_name>.json`.
- **1D:**  
  `python -m scripts.tune_1d --config configs/exp_1d_A1_20class.yaml`  
  Grouped runs use `configs/exp_1d_grouped_9.yaml`, which forces a 9-way output. The search uses the full 16384-sample waveform.

---

## 6. Train the best model

- After tuning, the **best 2D model** is already saved as `best_model_<experiment_name>.h5`. You can either:
  - **Use it as-is** for evaluation and plots, or  
  - **Retrain** the same architecture for more epochs (e.g. copy best_hps into a training config and run `train_experiment` with a higher `max_epochs`) then replace the saved weights.
- **2D (no tuning / fixed arch):**  
  `python -m scripts.train_experiment --config configs/exp_A2_20class_2d.yaml`
- **1D:**  
  `python -m scripts.train_1d_experiment --config configs/exp_1d_A1_20class.yaml`

---

## 7. Get results

- **2D:**  
  - `python -m scripts.evaluate_experiment --config configs/exp_A2_20class_2d.yaml`  
    → confusion matrix, training curves, SNR vs accuracy (if `snrs_logs/` present).  
  - `python -m scripts.analyze_features --config ...`  
    → t-SNE / PCA.  
  - `python -m scripts.calibrate_experiment --config ...`  
    → reliability diagram, ECE.
- **1D:**  
  - `python -m scripts.evaluate_1d_experiment --config configs/exp_1d_A1_20class.yaml`  
    → confusion matrix, training curves.

---

## Summary table

| Step | What you do | Script / location |
|------|-------------|--------------------|
| 1 | Receive raw `.txt` waveforms | — |
| 2 | Sort into `RAW_ROOT/<mechanism>/*.txt` | Your own script if needed |
| 3 | Produce `BASE_PATH/snrs_logs/*_snrs*.txt` (one float per line, order = log rows) | `scripts.build_snr_aggregate_from_log` (one run per paper; set `--snr-column`) |
| 4 | Build spectrograms for 2D | `scripts.build_2d_dataset_from_raw` |
| 5 | Tune 2D and 1D | `scripts.tune_2d`, `scripts.tune_1d` |
| 6 | Train (or retrain) best model | `scripts.train_experiment` / `train_1d_experiment` |
| 7 | Evaluate and plot | `evaluate_experiment`, `evaluate_1d_experiment`, `analyze_features`, `calibrate_experiment` |

All paths (e.g. `BASE_PATH`, `RAW_ROOT`) are set via config or environment so the same repo works on your machine or the cluster.
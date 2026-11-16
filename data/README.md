# Data Directory

This directory contains waveform data and intermediate datasets. **Do not commit files to this directory.**

## Structure

```
data/
├── raw/                 # Raw waveform files (*.txt, *.h5, etc.)
│   ├── Radice_2019/
│   ├── Powell_2020/
│   ├── Pan_2021/
│   ├── Mezzacappa_2020/
│   ├── Powell_2024/
│   ├── Choi_2024/
│   └── ...
├── processed/          # Resampled, normalized waveforms
└── spectrograms/       # Generated spectrograms for 2D CNN
```

## Downloading Data

Waveforms should be downloaded from:  
`C:\Users\ID611\CCSN Paper\waveforms repo\`

Use the download script:
```bash
python code/scripts/download_waveforms.py
```

Or create a symlink:
```bash
cd data/raw
ln -s "C:\Users\ID611\CCSN Paper\waveforms repo" waveforms
```

## Data Configuration

See `configs/waveform_selection.yaml` for which waveforms to use.

## Note

All files in this directory are ignored by git (see `.gitignore`). This is intentional because:
1. Waveforms are large (100s of MB)
2. Processed data can be regenerated from raw
3. Intermediate outputs should not be versioned
4. Data provenance is documented in configs

Always commit the **configuration** files, not the data itself.

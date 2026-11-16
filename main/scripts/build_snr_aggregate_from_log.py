"""
Build per-paper SNR aggregate files (*_snrs.txt) from a single log file.

Log format: one row per waveform. You must specify which column (0-based index)
contains the SNR so we can write one float per line in the same row order.
Use this when you have a log like Dir_main/100k/log files/log_mezzacappa2020.txt
and need BASE_PATH/snrs_logs/mezzacappa2020_snrs.txt for the pipeline.

Row order in the output matches the log file order. Ensure your waveform files
for this paper are named or listed in the same order (e.g. mezzacappa2020_0.txt,
mezzacappa2020_1.txt, ...) so that snr_dict[paper][index] matches.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build *_snrs.txt from a per-paper log (one SNR per line, same order as log rows)."
    )
    parser.add_argument("--log", required=True, help="Path to log file (e.g. log_mezzacappa2020.txt).")
    parser.add_argument("--output", required=True, help="Path to output file (e.g. snrs_logs/mezzacappa2020_snrs.txt).")
    parser.add_argument(
        "--snr-column",
        type=int,
        default=2,
        help="0-based column index of SNR in each space-separated line. Default 2.",
    )
    parser.add_argument("--sep", default=None, help="Column separator (default: split on whitespace).")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.output)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    snrs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(args.sep) if args.sep else line.split()
            if args.snr_column >= len(parts):
                raise ValueError(
                    f"Line has {len(parts)} columns; snr_column={args.snr_column}. First line: {line[:80]}..."
                )
            try:
                snrs.append(float(parts[args.snr_column]))
            except ValueError as e:
                raise ValueError(f"Could not parse SNR from column {args.snr_column}: {parts[args.snr_column]!r}") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in snrs:
            f.write(f"{s}\n")

    print(f"Wrote {len(snrs)} SNRs to {out_path}")


if __name__ == "__main__":
    main()

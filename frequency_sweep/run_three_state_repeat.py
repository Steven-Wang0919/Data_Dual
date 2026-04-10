from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def infer_target_freq(repeat_dir: Path) -> float:
    frequency_name = repeat_dir.parent.name
    match = re.search(r"(\d+)\s*Hz", frequency_name, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", frequency_name)
    if not match:
        raise ValueError(f"Unable to infer target frequency from directory name: {frequency_name}")
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-state soil-signal analysis for one repeat directory."
    )
    parser.add_argument("repeat_dir", help="Path to one repeat directory, for example 0300Hz/repeat_01.")
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--target-freq", type=float, default=None)
    parser.add_argument("--band-low", type=float, default=None)
    parser.add_argument("--band-high", type=float, default=None)
    parser.add_argument("--band-half-width", type=float, default=10.0)
    parser.add_argument("--transition-hz", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--detail-seconds", type=float, default=2.0)
    parser.add_argument("--representative-seconds", type=float, default=0.2)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repeat_dir = Path(args.repeat_dir).expanduser().resolve()
    if not repeat_dir.is_dir():
        raise FileNotFoundError(f"Repeat directory not found: {repeat_dir}")

    project_root = repeat_dir.parents[3]
    compare_script = project_root / "Data_Dual" / "frequency_sweep" / "compare_three_soil_signals.py"

    target_freq = float(args.target_freq) if args.target_freq is not None else infer_target_freq(repeat_dir)
    band_low = float(args.band_low) if args.band_low is not None else target_freq - args.band_half_width
    band_high = float(args.band_high) if args.band_high is not None else target_freq + args.band_half_width

    raw_dir = repeat_dir / "raw"
    output_dir = repeat_dir / "analysis" / "three_state_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    background_csv = raw_dir / "no_audio.csv"
    without_csv = raw_dir / "without_object.csv"
    with_csv = raw_dir / "with_object.csv"

    missing = [path.name for path in (background_csv, without_csv, with_csv) if not path.is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required raw files: {missing_text}")

    command = [
        sys.executable,
        str(compare_script),
        str(background_csv),
        str(without_csv),
        str(with_csv),
        "--sample-rate",
        f"{args.sample_rate}",
        "--target-freq",
        f"{target_freq}",
        "--band-low",
        f"{band_low}",
        "--band-high",
        f"{band_high}",
        "--transition-hz",
        f"{args.transition_hz}",
        "--window-size",
        f"{args.window_size}",
        "--step-size",
        f"{args.step_size}",
        "--detail-seconds",
        f"{args.detail_seconds}",
        "--representative-seconds",
        f"{args.representative_seconds}",
        "--settle-seconds",
        f"{args.settle_seconds}",
        "--output-dir",
        str(output_dir),
    ]

    print(f"Running {repeat_dir.parent.name}/{repeat_dir.name} three-state analysis...")
    print(f"Script: {compare_script}")
    print(f"Output: {output_dir}")
    subprocess.run(command, check=True)
    print("Analysis complete.")


if __name__ == "__main__":
    main()

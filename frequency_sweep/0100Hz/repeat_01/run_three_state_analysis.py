from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-state soil-signal analysis for frequency_sweep/0100Hz/repeat_01."
    )
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--target-freq", type=float, default=100.0)
    parser.add_argument("--band-low", type=float, default=90.0)
    parser.add_argument("--band-high", type=float, default=110.0)
    parser.add_argument("--transition-hz", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument(
        "--detail-seconds",
        type=float,
        default=2.0,
        help="Length of the representative detail window shown in three_state_filtered_waveform_detail.",
    )
    parser.add_argument(
        "--representative-seconds",
        type=float,
        default=0.2,
        help="Length of the short representative waveform window used in the representative plot.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Ignore the first and last part of each record when selecting representative segments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repeat_dir = Path(__file__).resolve().parent
    project_root = repeat_dir.parents[3]
    compare_script = project_root / "Python" / "compare_three_soil_signals.py"

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
        f"{args.target_freq}",
        "--band-low",
        f"{args.band_low}",
        "--band-high",
        f"{args.band_high}",
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

    print("Running repeat_01 three-state analysis...")
    print(f"Script: {compare_script}")
    print(f"Output: {output_dir}")
    subprocess.run(command, check=True)
    print("Analysis complete.")


if __name__ == "__main__":
    main()

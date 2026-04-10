from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "frequency_sweep" and SCRIPT_DIR.parent.name == "Data_Dual":
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_HELPER_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_HELPER_DIR))

from analyze_soil_signal import save_svg_line_plot, write_csv
from compare_soil_signals import summarize_signal


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parents[1] / "Data_Dual" / "100HZ实验组" / "串口数据"

    parser = argparse.ArgumentParser(
        description="Compare background, without-object, and with-object soil vibration CSV files."
    )
    parser.add_argument(
        "background_csv",
        nargs="?",
        default=str(default_dir / "无音频.csv"),
        help="CSV collected without audio excitation.",
    )
    parser.add_argument(
        "without_object_csv",
        nargs="?",
        default=str(default_dir / "100Hz无异物.csv"),
        help="CSV collected with 100Hz excitation and without foreign object.",
    )
    parser.add_argument(
        "with_object_csv",
        nargs="?",
        default=str(default_dir / "100Hz有异物.csv"),
        help="CSV collected with 100Hz excitation and with foreign object.",
    )
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--target-freq", type=float, default=100.0)
    parser.add_argument("--band-low", type=float, default=90.0)
    parser.add_argument("--band-high", type=float, default=110.0)
    parser.add_argument("--transition-hz", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--detail-seconds", type=float, default=2.0)
    parser.add_argument("--representative-seconds", type=float, default=0.2)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to store comparison outputs.",
    )
    return parser.parse_args()


def ratio(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return float("inf")
    return numerator / denominator


def select_representative_segment(
    data: dict[str, Any],
    sample_rate: float,
    segment_seconds: float,
    settle_seconds: float,
) -> dict[str, Any]:
    centered_signal = np.asarray(data["centered_signal"], dtype=np.float64)
    filtered_signal = np.asarray(data["filtered_signal"], dtype=np.float64)
    total_samples = filtered_signal.size
    segment_samples = max(2, int(round(segment_seconds * sample_rate)))
    segment_samples = min(segment_samples, total_samples)
    half_segment = segment_samples // 2
    total_duration = total_samples / sample_rate
    time_tolerance = max(1e-9, 0.5 / sample_rate)

    window_rows = list(data["window_rows"])
    eligible_windows = [
        row
        for row in window_rows
        if row["start_time_s"] >= settle_seconds - time_tolerance
        and row["end_time_s"] <= max(settle_seconds, total_duration - settle_seconds) + time_tolerance
    ]
    if not eligible_windows:
        eligible_windows = window_rows

    if eligible_windows:
        amplitudes = np.asarray(
            [row["filtered_target_amp"] for row in eligible_windows], dtype=np.float64
        )
        median_amplitude = float(np.median(amplitudes))
        selected_index = int(np.argmin(np.abs(amplitudes - median_amplitude)))
        selected_window = eligible_windows[selected_index]
        center_time = 0.5 * (
            float(selected_window["start_time_s"]) + float(selected_window["end_time_s"])
        )
        selected_amplitude = float(selected_window["filtered_target_amp"])
    else:
        center_time = max(segment_seconds * 0.5, total_duration * 0.5)
        selected_amplitude = 0.0

    center_index = int(round(center_time * sample_rate))
    start_index = max(0, min(center_index - half_segment, total_samples - segment_samples))
    end_index = start_index + segment_samples

    local_time = np.arange(end_index - start_index, dtype=np.float64) / sample_rate
    centered_segment = centered_signal[start_index:end_index]
    filtered_segment = filtered_signal[start_index:end_index]

    return {
        "time": local_time,
        "centered_signal": centered_segment,
        "filtered_signal": filtered_segment,
        "start_time_s": start_index / sample_rate,
        "end_time_s": end_index / sample_rate,
        "center_time_s": center_time,
        "segment_samples": int(end_index - start_index),
        "window_target_amp": selected_amplitude,
    }


def build_segment_candidates(
    data: dict[str, Any],
    sample_rate: float,
    segment_seconds: float,
    settle_seconds: float,
    step_size: int,
) -> list[dict[str, float]]:
    filtered_signal = np.asarray(data["filtered_signal"], dtype=np.float64)
    total_samples = filtered_signal.size
    total_duration = total_samples / sample_rate
    segment_samples = max(2, int(round(segment_seconds * sample_rate)))
    segment_samples = min(segment_samples, total_samples)

    window_rows = list(data["window_rows"])
    if not window_rows:
        return []

    earliest_start = max(0.0, settle_seconds)
    latest_start = max(0.0, total_duration - settle_seconds - segment_seconds)
    if latest_start < earliest_start:
        earliest_start = 0.0
        latest_start = max(0.0, total_duration - segment_seconds)

    candidate_step_s = max(step_size / sample_rate, 0.05)
    candidate_starts = np.arange(earliest_start, latest_start + candidate_step_s * 0.5, candidate_step_s)
    if candidate_starts.size == 0:
        candidate_starts = np.asarray([max(0.0, (total_duration - segment_seconds) * 0.5)], dtype=np.float64)
    time_tolerance = max(1e-9, 0.5 / sample_rate)

    all_amps = np.asarray([row["filtered_target_amp"] for row in window_rows], dtype=np.float64)
    global_median_amp = float(np.median(all_amps))
    global_scale = max(global_median_amp, float(np.std(all_amps)), 1e-6)
    row_duration_s = max(
        1.0 / sample_rate,
        float(window_rows[0]["end_time_s"]) - float(window_rows[0]["start_time_s"]),
    )
    if segment_seconds <= row_duration_s + 1e-9:
        min_rows = 1
    else:
        min_rows = max(2, int(round(segment_seconds * sample_rate / max(step_size, 1))) // 2)

    candidates: list[dict[str, float]] = []
    for candidate_start in candidate_starts:
        candidate_end = float(candidate_start + segment_seconds)
        rows = [
            row
            for row in window_rows
            if row["start_time_s"] >= candidate_start - time_tolerance
            and row["end_time_s"] <= candidate_end + time_tolerance
        ]
        if len(rows) < min_rows:
            continue

        amps = np.asarray([row["filtered_target_amp"] for row in rows], dtype=np.float64)
        mean_amp = float(np.mean(amps))
        std_amp = float(np.std(amps))
        stability_score = std_amp / max(mean_amp, global_scale, 1e-6)
        typical_penalty = abs(mean_amp - global_median_amp) / global_scale
        candidates.append(
            {
                "start_time_s": float(candidate_start),
                "end_time_s": candidate_end,
                "mean_amp": mean_amp,
                "std_amp": std_amp,
                "global_scale": global_scale,
                "stability_score": stability_score,
                "typical_penalty": typical_penalty,
            }
        )

    return candidates


def materialize_segment(
    data: dict[str, Any],
    sample_rate: float,
    segment_seconds: float,
    candidate: dict[str, float],
) -> dict[str, Any]:
    centered_signal = np.asarray(data["centered_signal"], dtype=np.float64)
    filtered_signal = np.asarray(data["filtered_signal"], dtype=np.float64)
    total_samples = filtered_signal.size
    segment_samples = max(2, int(round(segment_seconds * sample_rate)))
    segment_samples = min(segment_samples, total_samples)

    start_index = int(round(candidate["start_time_s"] * sample_rate))
    start_index = max(0, min(start_index, total_samples - segment_samples))
    end_index = start_index + segment_samples
    local_time = np.arange(end_index - start_index, dtype=np.float64) / sample_rate

    return {
        "time": local_time,
        "centered_signal": centered_signal[start_index:end_index],
        "filtered_signal": filtered_signal[start_index:end_index],
        "start_time_s": start_index / sample_rate,
        "end_time_s": end_index / sample_rate,
        "center_time_s": 0.5 * ((start_index / sample_rate) + (end_index / sample_rate)),
        "segment_samples": int(end_index - start_index),
        "window_target_amp": candidate["mean_amp"],
        "window_target_std": candidate.get("std_amp", 0.0),
    }


def select_stable_segment(
    data: dict[str, Any],
    sample_rate: float,
    segment_seconds: float,
    settle_seconds: float,
    step_size: int,
) -> dict[str, Any]:
    candidates = build_segment_candidates(data, sample_rate, segment_seconds, settle_seconds, step_size)
    if not candidates:
        return select_representative_segment(data, sample_rate, segment_seconds, settle_seconds)

    best_choice = min(
        candidates,
        key=lambda candidate: candidate["stability_score"] + 0.35 * candidate["typical_penalty"],
    )
    return materialize_segment(data, sample_rate, segment_seconds, best_choice)


def select_contrast_segments(
    without_data: dict[str, Any],
    with_data: dict[str, Any],
    sample_rate: float,
    segment_seconds: float,
    settle_seconds: float,
    step_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    without_candidates = build_segment_candidates(
        without_data,
        sample_rate,
        segment_seconds,
        settle_seconds,
        step_size,
    )
    with_candidates = build_segment_candidates(
        with_data,
        sample_rate,
        segment_seconds,
        settle_seconds,
        step_size,
    )

    if not without_candidates or not with_candidates:
        return (
            select_stable_segment(without_data, sample_rate, segment_seconds, settle_seconds, step_size),
            select_stable_segment(with_data, sample_rate, segment_seconds, settle_seconds, step_size),
        )

    best_pair: tuple[dict[str, float], dict[str, float]] | None = None
    best_score = float("-inf")

    is_short_segment = segment_seconds <= 0.3

    for without_candidate in without_candidates:
        for with_candidate in with_candidates:
            without_amp = without_candidate["mean_amp"]
            with_amp = with_candidate["mean_amp"]
            contrast_scale = max(
                without_candidate["global_scale"],
                with_candidate["global_scale"],
                1e-6,
            )
            log_ratio = float(np.log((without_amp + 1e-6) / (with_amp + 1e-6)))
            normalized_gap = (without_amp - with_amp) / contrast_scale
            penalty = (
                (0.45 if is_short_segment else 0.7)
                * (without_candidate["stability_score"] + with_candidate["stability_score"])
                + (0.12 if is_short_segment else 0.2)
                * (without_candidate["typical_penalty"] + with_candidate["typical_penalty"])
            )
            score = (
                (1.4 if is_short_segment else 1.0) * log_ratio
                + (1.2 if is_short_segment else 0.8) * normalized_gap
                + (0.35 if is_short_segment else 0.15) * (without_amp / contrast_scale)
                - penalty
            )

            if without_amp <= with_amp:
                score -= (2.4 if is_short_segment else 1.5) * (with_amp - without_amp) / contrast_scale

            if score > best_score:
                best_score = score
                best_pair = (without_candidate, with_candidate)

    if best_pair is None:
        return (
            select_stable_segment(without_data, sample_rate, segment_seconds, settle_seconds, step_size),
            select_stable_segment(with_data, sample_rate, segment_seconds, settle_seconds, step_size),
        )

    return (
        materialize_segment(without_data, sample_rate, segment_seconds, best_pair[0]),
        materialize_segment(with_data, sample_rate, segment_seconds, best_pair[1]),
    )


def render_metric_rows(
    background_summary: dict[str, Any],
    without_summary: dict[str, Any],
    with_summary: dict[str, Any],
    metrics: list[str],
) -> list[tuple[str, float, float, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for metric in metrics:
        background_value = float(background_summary[metric])
        without_value = float(without_summary[metric])
        with_value = float(with_summary[metric])
        rows.append(
            (
                metric,
                background_value,
                without_value,
                with_value,
                ratio(without_value, background_value),
                ratio(with_value, without_value),
                ratio(with_value, background_value),
            )
        )
    return rows


def save_report(
    output_dir: Path,
    background_data: dict[str, Any],
    without_data: dict[str, Any],
    with_data: dict[str, Any],
    metric_rows: list[tuple[str, float, float, float, float, float, float]],
    plot_files: list[str],
) -> None:
    table_rows = []
    for metric, background_value, without_value, with_value, without_bg, with_without, with_bg in metric_rows:
        table_rows.append(
            "<tr>"
            f"<td>{escape(metric)}</td>"
            f"<td>{background_value:.6f}</td>"
            f"<td>{without_value:.6f}</td>"
            f"<td>{with_value:.6f}</td>"
            f"<td>{'inf' if not np.isfinite(without_bg) else f'{without_bg:.4f}'}</td>"
            f"<td>{'inf' if not np.isfinite(with_without) else f'{with_without:.4f}'}</td>"
            f"<td>{'inf' if not np.isfinite(with_bg) else f'{with_bg:.4f}'}</td>"
            "</tr>"
        )

    images = "\n".join(
        f'<h2>{escape(name)}</h2><img src="{escape(name)}" style="max-width:100%; border:1px solid #d1d5db;" />'
        for name in plot_files
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Three-State Soil Signal Comparison Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    h1, h2 {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>Three-State Soil Signal Comparison Report</h1>
  <p>Background: {escape(str(background_data["path"]))}</p>
  <p>Without object: {escape(str(without_data["path"]))}</p>
  <p>With object: {escape(str(with_data["path"]))}</p>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>无音频</th>
        <th>100Hz无异物</th>
        <th>100Hz有异物</th>
        <th>无异物/无音频</th>
        <th>有异物/无异物</th>
        <th>有异物/无音频</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
  {images}
</body>
</html>
"""
    (output_dir / "three_state_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    background_path = Path(args.background_csv).expanduser().resolve()
    without_path = Path(args.without_object_csv).expanduser().resolve()
    with_path = Path(args.with_object_csv).expanduser().resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else background_path.parent / "三组对比分析"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    background_data = summarize_signal(
        background_path,
        args.sample_rate,
        args.target_freq,
        args.band_low,
        args.band_high,
        args.transition_hz,
        args.window_size,
        args.step_size,
    )
    without_data = summarize_signal(
        without_path,
        args.sample_rate,
        args.target_freq,
        args.band_low,
        args.band_high,
        args.transition_hz,
        args.window_size,
        args.step_size,
    )
    with_data = summarize_signal(
        with_path,
        args.sample_rate,
        args.target_freq,
        args.band_low,
        args.band_high,
        args.transition_hz,
        args.window_size,
        args.step_size,
    )

    metrics = [
        "centered_rms",
        "raw_target_amp",
        "filtered_rms",
        "filtered_target_amp",
        "filtered_energy_ratio",
        "window_filtered_target_amp_mean",
        "window_filtered_target_amp_std",
    ]
    metric_rows = render_metric_rows(
        background_data["summary"],
        without_data["summary"],
        with_data["summary"],
        metrics,
    )

    write_csv(
        output_dir / "three_state_metrics.csv",
        (
            "metric",
            "background_no_audio",
            "without_object",
            "with_object",
            "without_div_background",
            "with_div_without",
            "with_div_background",
        ),
        metric_rows,
    )

    min_len = min(
        background_data["time_axis"].size,
        without_data["time_axis"].size,
        with_data["time_axis"].size,
    )
    write_csv(
        output_dir / "three_state_waveforms.csv",
        (
            "time_s",
            "background_centered_adc",
            "without_centered_adc",
            "with_centered_adc",
            "background_filtered_100hz_adc",
            "without_filtered_100hz_adc",
            "with_filtered_100hz_adc",
        ),
        zip(
            background_data["time_axis"][:min_len],
            background_data["centered_signal"][:min_len],
            without_data["centered_signal"][:min_len],
            with_data["centered_signal"][:min_len],
            background_data["filtered_signal"][:min_len],
            without_data["filtered_signal"][:min_len],
            with_data["filtered_signal"][:min_len],
        ),
    )

    background_windows = background_data["window_rows"]
    without_windows = without_data["window_rows"]
    with_windows = with_data["window_rows"]

    spectrum_limit = min(
        500.0,
        float(background_data["freqs"][-1]) if background_data["freqs"].size else 500.0,
        float(without_data["freqs"][-1]) if without_data["freqs"].size else 500.0,
        float(with_data["freqs"][-1]) if with_data["freqs"].size else 500.0,
    )
    bg_limit_idx = np.searchsorted(background_data["freqs"], spectrum_limit, side="right")
    without_limit_idx = np.searchsorted(without_data["freqs"], spectrum_limit, side="right")
    with_limit_idx = np.searchsorted(with_data["freqs"], spectrum_limit, side="right")

    background_segment = select_representative_segment(
        background_data,
        args.sample_rate,
        args.representative_seconds,
        args.settle_seconds,
    )
    without_segment, with_segment = select_contrast_segments(
        without_data,
        with_data,
        args.sample_rate,
        args.representative_seconds,
        args.settle_seconds,
        args.step_size,
    )
    background_detail_segment = select_stable_segment(
        background_data,
        args.sample_rate,
        args.detail_seconds,
        args.settle_seconds,
        args.step_size,
    )
    without_detail_segment, with_detail_segment = select_contrast_segments(
        without_data,
        with_data,
        args.sample_rate,
        args.detail_seconds,
        args.settle_seconds,
        args.step_size,
    )

    target_focus_low = max(0.0, args.target_freq - max(40.0, args.target_freq * 0.4))
    target_focus_high = min(spectrum_limit, args.target_freq + max(120.0, args.target_freq * 1.5))

    write_csv(
        output_dir / "three_state_representative_windows.csv",
        (
            "label",
            "selected_start_s",
            "selected_end_s",
            "selected_center_s",
            "selected_window_target_amp",
            "segment_samples",
        ),
        [
            (
                "no_audio",
                background_segment["start_time_s"],
                background_segment["end_time_s"],
                background_segment["center_time_s"],
                background_segment["window_target_amp"],
                background_segment["segment_samples"],
            ),
            (
                "without_object",
                without_segment["start_time_s"],
                without_segment["end_time_s"],
                without_segment["center_time_s"],
                without_segment["window_target_amp"],
                without_segment["segment_samples"],
            ),
            (
                "with_object",
                with_segment["start_time_s"],
                with_segment["end_time_s"],
                with_segment["center_time_s"],
                with_segment["window_target_amp"],
                with_segment["segment_samples"],
            ),
        ],
    )

    write_csv(
        output_dir / "three_state_detail_windows.csv",
        (
            "label",
            "selected_start_s",
            "selected_end_s",
            "selected_center_s",
            "selected_window_target_amp",
            "selected_window_target_std",
            "segment_samples",
        ),
        [
            (
                "no_audio",
                background_detail_segment["start_time_s"],
                background_detail_segment["end_time_s"],
                background_detail_segment["center_time_s"],
                background_detail_segment["window_target_amp"],
                background_detail_segment.get("window_target_std", 0.0),
                background_detail_segment["segment_samples"],
            ),
            (
                "without_object",
                without_detail_segment["start_time_s"],
                without_detail_segment["end_time_s"],
                without_detail_segment["center_time_s"],
                without_detail_segment["window_target_amp"],
                without_detail_segment.get("window_target_std", 0.0),
                without_detail_segment["segment_samples"],
            ),
            (
                "with_object",
                with_detail_segment["start_time_s"],
                with_detail_segment["end_time_s"],
                with_detail_segment["center_time_s"],
                with_detail_segment["window_target_amp"],
                with_detail_segment.get("window_target_std", 0.0),
                with_detail_segment["segment_samples"],
            ),
        ],
    )

    plot_files: list[str] = []

    filtered_svg = output_dir / "three_state_filtered_waveform_detail.svg"
    save_svg_line_plot(
        filtered_svg,
        f"100Hz Band-Pass Detail (Representative {args.detail_seconds:.1f}s)",
        "Time Within Selected Window (s)",
        "ADC",
        [
            {
                "x": background_detail_segment["time"],
                "y": background_detail_segment["filtered_signal"],
                "label": "no audio",
                "color": "#6b7280",
            },
            {
                "x": without_detail_segment["time"],
                "y": without_detail_segment["filtered_signal"],
                "label": "without object",
                "color": "#2563eb",
            },
            {
                "x": with_detail_segment["time"],
                "y": with_detail_segment["filtered_signal"],
                "label": "with object",
                "color": "#dc2626",
            },
        ],
        x_limit=(0.0, args.detail_seconds),
    )
    plot_files.append(filtered_svg.name)

    spectrum_svg = output_dir / "three_state_spectrum.svg"
    save_svg_line_plot(
        spectrum_svg,
        "Object-Focused Spectrum Comparison",
        "Frequency (Hz)",
        "Amplitude",
        [
            {
                "x": background_data["freqs"][:bg_limit_idx],
                "y": background_data["raw_amplitudes"][:bg_limit_idx],
                "label": "no audio (reference)",
                "color": "#9ca3af",
            },
            {
                "x": without_data["freqs"][:without_limit_idx],
                "y": without_data["raw_amplitudes"][:without_limit_idx],
                "label": "without object",
                "color": "#2563eb",
            },
            {
                "x": with_data["freqs"][:with_limit_idx],
                "y": with_data["raw_amplitudes"][:with_limit_idx],
                "label": "with object",
                "color": "#dc2626",
            },
        ],
        band=(args.band_low, args.band_high),
        vlines=[(args.target_freq, "#7c3aed", f"target {args.target_freq:.1f}Hz")],
        x_limit=(target_focus_low, target_focus_high),
    )
    plot_files.append(spectrum_svg.name)

    representative_svg = output_dir / "three_state_representative_filtered_waveform.svg"
    save_svg_line_plot(
        representative_svg,
        f"Representative 100Hz Waveform ({args.representative_seconds:.2f}s, Reference Included)",
        "Time Within Selected Window (s)",
        "ADC",
        [
            {
                "x": background_segment["time"],
                "y": background_segment["filtered_signal"],
                "label": "no audio (reference)",
                "color": "#9ca3af",
            },
            {
                "x": without_segment["time"],
                "y": without_segment["filtered_signal"],
                "label": "without object",
                "color": "#2563eb",
            },
            {
                "x": with_segment["time"],
                "y": with_segment["filtered_signal"],
                "label": "with object",
                "color": "#dc2626",
            },
        ],
        x_limit=(0.0, args.representative_seconds),
    )
    plot_files.append(representative_svg.name)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            background_detail_segment["time"],
            background_detail_segment["filtered_signal"],
            label="no audio",
            linewidth=1.1,
            color="#6b7280",
        )
        ax.plot(
            without_detail_segment["time"],
            without_detail_segment["filtered_signal"],
            label="without object",
            linewidth=1.1,
            color="#2563eb",
        )
        ax.plot(
            with_detail_segment["time"],
            with_detail_segment["filtered_signal"],
            label="with object",
            linewidth=1.1,
            color="#dc2626",
        )
        ax.set_xlim(0.0, args.detail_seconds)
        ax.set_xlabel("Time Within Selected Window (s)")
        ax.set_ylabel("ADC")
        ax.set_title(
            f"100Hz Band-Pass Detail (Representative {args.detail_seconds:.1f}s)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "three_state_filtered_waveform_detail.png", dpi=150)
        plt.close(fig)
        plot_files.append("three_state_filtered_waveform_detail.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            background_data["freqs"][:bg_limit_idx],
            background_data["raw_amplitudes"][:bg_limit_idx],
            label="no audio (reference)",
            linewidth=0.9,
            color="#9ca3af",
        )
        ax.plot(
            without_data["freqs"][:without_limit_idx],
            without_data["raw_amplitudes"][:without_limit_idx],
            label="without object",
            linewidth=0.9,
            color="#2563eb",
        )
        ax.plot(
            with_data["freqs"][:with_limit_idx],
            with_data["raw_amplitudes"][:with_limit_idx],
            label="with object",
            linewidth=0.9,
            color="#dc2626",
        )
        ax.axvspan(args.band_low, args.band_high, color="#fde68a", alpha=0.35)
        ax.axvline(args.target_freq, color="#7c3aed", linestyle="--", linewidth=1.0)
        ax.set_xlim(target_focus_low, target_focus_high)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Object-Focused Spectrum Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "three_state_spectrum.png", dpi=150)
        plt.close(fig)
        plot_files.append("three_state_spectrum.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            background_segment["time"],
            background_segment["filtered_signal"],
            label="no audio (reference)",
            linewidth=1.1,
            color="#9ca3af",
        )
        ax.plot(
            without_segment["time"],
            without_segment["filtered_signal"],
            label="without object",
            linewidth=1.1,
            color="#2563eb",
        )
        ax.plot(
            with_segment["time"],
            with_segment["filtered_signal"],
            label="with object",
            linewidth=1.1,
            color="#dc2626",
        )
        ax.set_xlim(0.0, args.representative_seconds)
        ax.set_xlabel("Time Within Selected Window (s)")
        ax.set_ylabel("ADC")
        ax.set_title(
            f"Representative 100Hz Waveform ({args.representative_seconds:.2f}s, Reference Included)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "three_state_representative_filtered_waveform.png", dpi=150)
        plt.close(fig)
        plot_files.append("three_state_representative_filtered_waveform.png")

    save_report(output_dir, background_data, without_data, with_data, metric_rows, plot_files)

    with (output_dir / "background_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(background_data["summary"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "without_object_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(without_data["summary"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "with_object_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(with_data["summary"], handle, ensure_ascii=False, indent=2)

    print("Three-state comparison complete.")
    print(f"Output directory: {output_dir}")
    for metric, background_value, without_value, with_value, without_bg, with_without, with_bg in metric_rows:
        print(
            f"{metric}: "
            f"background={background_value:.6f}, "
            f"without={without_value:.6f}, "
            f"with={with_value:.6f}, "
            f"without/background={'inf' if not np.isfinite(without_bg) else f'{without_bg:.4f}'}, "
            f"with/without={'inf' if not np.isfinite(with_without) else f'{with_without:.4f}'}, "
            f"with/background={'inf' if not np.isfinite(with_bg) else f'{with_bg:.4f}'}"
        )


if __name__ == "__main__":
    main()

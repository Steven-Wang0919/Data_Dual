from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FREQUENCY_PATTERN = re.compile(r"^(\d+)\s*Hz$", flags=re.IGNORECASE)
REPEAT_PATTERN = re.compile(r"^repeat_(\d+)$", flags=re.IGNORECASE)
RAW_SIGNAL_FILES = {
    "no_audio": "no_audio.csv",
    "without_object": "without_object.csv",
    "with_object": "with_object.csv",
}
SERIES_STYLE = {
    "no_audio": {"label": "no audio (reference)", "color": "#9ca3af", "linewidth": 1.2, "alpha": 0.95},
    "without_object": {"label": "without object", "color": "#2563eb", "linewidth": 1.25, "alpha": 0.98},
    "with_object": {"label": "with object", "color": "#dc2626", "linewidth": 1.25, "alpha": 0.98},
}
FIGSIZE = (12, 4)


@dataclass(frozen=True)
class CandidateWindow:
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    center_s: float
    rms: float
    envelope_cv: float
    target_amp: float
    score: float
    center_distance_s: float


@dataclass(frozen=True)
class SelectedWindow:
    state_key: str
    rank: int
    candidate: CandidateWindow


@dataclass(frozen=True)
class AggregatedMetrics:
    target_amp_median: float
    target_amp_iqr: float
    envelope_cv_median: float
    envelope_cv_iqr: float


@dataclass(frozen=True)
class AlignedSegment:
    state_key: str
    start_s: float
    end_s: float
    center_s: float
    sample_rate: float
    segment_samples: int
    target_amp: float
    phase_rad: float
    phase_shift_s: float
    time_s: np.ndarray
    x_axis: np.ndarray
    raw_segment: np.ndarray
    aligned_segment: np.ndarray


@dataclass(frozen=True)
class PlotSpec:
    output_name: str
    duration_mode: str
    duration_value: float
    x_label: str
    title_template: str


PLOT_SPECS = (
    PlotSpec(
        output_name="three_state_phase_aligned_waveform",
        duration_mode="seconds",
        duration_value=0.2,
        x_label="Time Within Phase-Aligned Window (s)",
        title_template="Phase-Aligned {freq_label} Waveform",
    ),
    PlotSpec(
        output_name="three_state_phase_aligned_fixed_cycles_waveform",
        duration_mode="cycles",
        duration_value=5.0,
        x_label="Cycle Within Phase-Aligned Window",
        title_template="Phase-Aligned {freq_label} Fixed-Cycles Waveform",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate phase-aligned three-state comparison plots outside frequency_sweep."
    )
    parser.add_argument(
        "--frequency-root",
        type=Path,
        default=SCRIPT_DIR / "frequency_sweep",
        help="Root directory that contains 0100Hz/repeat_01 style experiment folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "Out_Pic_PhaseAligned",
        help="Directory used to store all new phase-aligned outputs.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        default=None,
        help="Optional frequency directory names to process, such as 0100Hz 0200Hz.",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Optional repeat directory names to process, such as repeat_01 repeat_02.",
    )
    parser.add_argument(
        "--sample-rate-multiplier",
        type=float,
        default=10.0,
        help="Nominal sample rate rule applied as sample_rate = target_freq * multiplier.",
    )
    parser.add_argument(
        "--band-half-width",
        type=float,
        default=10.0,
        help="Half-width of the target-frequency band-pass filter.",
    )
    parser.add_argument(
        "--transition-hz",
        type=float,
        default=5.0,
        help="Transition width used by the cosine-tapered band-pass filter.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Prefer windows that stay away from the first and last settle region.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of non-overlapping candidate windows retained before choosing a representative pair.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def infer_target_freq(frequency_name: str) -> float:
    match = FREQUENCY_PATTERN.match(frequency_name)
    if match is None:
        raise ValueError(f"Unable to infer target frequency from directory name: {frequency_name}")
    return float(match.group(1))


def format_frequency_label(target_freq: float) -> str:
    rounded = round(target_freq)
    if np.isclose(target_freq, rounded):
        return f"{int(rounded)}Hz"
    return f"{target_freq:g}Hz"


def frequency_sort_key(path: Path) -> float:
    return infer_target_freq(path.name)


def repeat_sort_key(path: Path) -> int:
    match = REPEAT_PATTERN.match(path.name)
    return int(match.group(1)) if match else 10**9


def read_single_column_signal(csv_path: Path) -> np.ndarray:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[0]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric samples found in {csv_path}")
    signal = np.asarray(values, dtype=np.float64)
    return signal - np.mean(signal)


def build_bandpass_response(freqs: np.ndarray, band_low: float, band_high: float, transition_hz: float) -> np.ndarray:
    response = np.zeros_like(freqs, dtype=np.float64)
    response[(freqs >= band_low) & (freqs <= band_high)] = 1.0

    if transition_hz <= 0.0:
        return response

    left_start = max(0.0, band_low - transition_hz)
    left_mask = (freqs >= left_start) & (freqs < band_low)
    if np.any(left_mask):
        position = (freqs[left_mask] - left_start) / transition_hz
        response[left_mask] = 0.5 - 0.5 * np.cos(np.pi * position)

    right_end = band_high + transition_hz
    right_mask = (freqs > band_high) & (freqs <= right_end)
    if np.any(right_mask):
        position = (freqs[right_mask] - band_high) / transition_hz
        response[right_mask] = 0.5 + 0.5 * np.cos(np.pi * position)

    return response


def apply_bandpass_filter(signal: np.ndarray, sample_rate: float, band_low: float, band_high: float, transition_hz: float) -> np.ndarray:
    if signal.size == 0:
        return signal.copy()
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    response = build_bandpass_response(freqs, band_low, band_high, transition_hz)
    filtered_fft = np.fft.rfft(signal) * response
    return np.fft.irfft(filtered_fft, n=signal.size)


def fit_target_component(signal: np.ndarray, sample_rate: float, target_freq: float) -> tuple[float, float]:
    if signal.size == 0:
        return 0.0, 0.0
    time_s = np.arange(signal.size, dtype=np.float64) / sample_rate
    omega_t = 2.0 * np.pi * target_freq * time_s
    design = np.column_stack([np.cos(omega_t), np.sin(omega_t)])
    coeffs, *_ = np.linalg.lstsq(design, signal, rcond=None)
    cos_coeff = float(coeffs[0])
    sin_coeff = float(coeffs[1])
    amplitude = float(math.hypot(cos_coeff, sin_coeff))
    phase_rad = math.atan2(-sin_coeff, cos_coeff)
    return amplitude, wrap_to_pi(phase_rad)


def wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def compute_window_seconds(plot_spec: PlotSpec, target_freq: float) -> float:
    if plot_spec.duration_mode == "seconds":
        return plot_spec.duration_value
    return plot_spec.duration_value / max(target_freq, 1e-9)


def build_candidate_windows(
    filtered_signal: np.ndarray,
    sample_rate: float,
    target_freq: float,
    segment_seconds: float,
    settle_seconds: float,
) -> list[CandidateWindow]:
    total_samples = filtered_signal.size
    if total_samples == 0:
        return []

    segment_samples = max(2, int(round(segment_seconds * sample_rate)))
    segment_samples = min(segment_samples, total_samples)
    settle_samples = max(0, int(round(settle_seconds * sample_rate)))
    max_start_index = total_samples - segment_samples

    earliest_start = settle_samples
    latest_start = max_start_index - settle_samples
    if latest_start < earliest_start:
        earliest_start = 0
        latest_start = max_start_index
    if latest_start < 0:
        return []

    samples_per_cycle = sample_rate / max(target_freq, 1e-9)
    step_samples = max(1, int(round(samples_per_cycle * 0.5)))
    start_indices = list(range(earliest_start, latest_start + 1, step_samples))
    if not start_indices or start_indices[-1] != latest_start:
        start_indices.append(latest_start)

    total_center_s = (total_samples / sample_rate) * 0.5
    candidates: list[CandidateWindow] = []
    for start_index in start_indices:
        end_index = start_index + segment_samples
        segment = filtered_signal[start_index:end_index]
        target_amp, _ = fit_target_component(segment, sample_rate, target_freq)
        rms = float(np.sqrt(np.mean(np.square(segment))))
        envelope = np.abs(segment)
        envelope_mean = max(float(np.mean(envelope)), 1e-12)
        envelope_cv = float(np.std(envelope) / envelope_mean)
        start_s = start_index / sample_rate
        end_s = end_index / sample_rate
        center_s = 0.5 * (start_s + end_s)
        score = rms / (1.0 + envelope_cv)
        candidates.append(
            CandidateWindow(
                start_index=start_index,
                end_index=end_index,
                start_s=start_s,
                end_s=end_s,
                center_s=center_s,
                rms=rms,
                envelope_cv=envelope_cv,
                target_amp=target_amp,
                score=score,
                center_distance_s=abs(center_s - total_center_s),
            )
        )
    return candidates


def pick_non_overlapping_candidates(
    sorted_candidates: Iterable[CandidateWindow],
    top_k: int,
    segment_samples: int,
) -> list[CandidateWindow]:
    spacing_options = [
        max(segment_samples, 1),
        max(segment_samples // 2, 1),
        1,
    ]
    best_selection: list[CandidateWindow] = []
    for min_spacing in spacing_options:
        selected: list[CandidateWindow] = []
        selected_centers: list[int] = []
        for candidate in sorted_candidates:
            center_index = (candidate.start_index + candidate.end_index) // 2
            if all(abs(center_index - existing_center) >= min_spacing for existing_center in selected_centers):
                selected.append(candidate)
                selected_centers.append(center_index)
            if len(selected) >= top_k:
                break
        if len(selected) > len(best_selection):
            best_selection = selected
        if len(best_selection) >= top_k:
            break
    return best_selection


def select_quiet_windows(candidates: list[CandidateWindow], top_k: int, segment_samples: int) -> list[SelectedWindow]:
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.rms,
            candidate.target_amp,
            candidate.envelope_cv,
            candidate.center_distance_s,
        ),
    )
    chosen = pick_non_overlapping_candidates(sorted_candidates, top_k, segment_samples)
    return [SelectedWindow(state_key="no_audio", rank=index, candidate=candidate) for index, candidate in enumerate(chosen, start=1)]


def select_signal_windows(state_key: str, candidates: list[CandidateWindow], top_k: int, segment_samples: int) -> list[SelectedWindow]:
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.target_amp,
            -candidate.envelope_cv,
            -candidate.center_distance_s,
        ),
        reverse=True,
    )
    chosen = pick_non_overlapping_candidates(sorted_candidates, top_k, segment_samples)
    return [SelectedWindow(state_key=state_key, rank=index, candidate=candidate) for index, candidate in enumerate(chosen, start=1)]


def aggregate_selected_windows(selected: list[SelectedWindow]) -> AggregatedMetrics:
    amps = np.asarray([item.candidate.target_amp for item in selected], dtype=np.float64)
    envs = np.asarray([item.candidate.envelope_cv for item in selected], dtype=np.float64)
    amp_q1, amp_q3 = np.percentile(amps, [25.0, 75.0]) if amps.size else (0.0, 0.0)
    env_q1, env_q3 = np.percentile(envs, [25.0, 75.0]) if envs.size else (0.0, 0.0)
    return AggregatedMetrics(
        target_amp_median=float(np.median(amps)) if amps.size else 0.0,
        target_amp_iqr=float(amp_q3 - amp_q1),
        envelope_cv_median=float(np.median(envs)) if envs.size else 0.0,
        envelope_cv_iqr=float(env_q3 - env_q1),
    )


def choose_representative_window(selected: list[SelectedWindow], metrics: AggregatedMetrics) -> SelectedWindow:
    if not selected:
        raise ValueError("Cannot choose a representative window from an empty selection.")
    if len(selected) == 1:
        return selected[0]

    amp_scale = max(metrics.target_amp_iqr, 0.05 * metrics.target_amp_median, 1e-6)
    env_scale = max(metrics.envelope_cv_iqr, 0.05 * metrics.envelope_cv_median, 1e-6)
    return min(
        selected,
        key=lambda item: (
            abs(item.candidate.target_amp - metrics.target_amp_median) / amp_scale
            + 0.45 * abs(item.candidate.envelope_cv - metrics.envelope_cv_median) / env_scale
            + 0.10 * max(item.rank - 1, 0)
        ),
    )


def choose_representative_pair(
    without_selected: list[SelectedWindow],
    with_selected: list[SelectedWindow],
) -> tuple[SelectedWindow, SelectedWindow]:
    if not without_selected or not with_selected:
        raise ValueError("Cannot choose a representative pair from empty selections.")

    without_metrics = aggregate_selected_windows(without_selected)
    with_metrics = aggregate_selected_windows(with_selected)
    gap_target = without_metrics.target_amp_median - with_metrics.target_amp_median
    gap_scale = max(
        without_metrics.target_amp_iqr + with_metrics.target_amp_iqr,
        0.05 * (without_metrics.target_amp_median + with_metrics.target_amp_median),
        1e-6,
    )

    def represent_penalty(item: SelectedWindow, metrics: AggregatedMetrics) -> float:
        amp_scale = max(metrics.target_amp_iqr, 0.05 * metrics.target_amp_median, 1e-6)
        env_scale = max(metrics.envelope_cv_iqr, 0.05 * metrics.envelope_cv_median, 1e-6)
        return (
            abs(item.candidate.target_amp - metrics.target_amp_median) / amp_scale
            + 0.45 * abs(item.candidate.envelope_cv - metrics.envelope_cv_median) / env_scale
            + 0.10 * max(item.rank - 1, 0)
        )

    best_row: tuple[float, float, SelectedWindow, SelectedWindow] | None = None
    for without_item in without_selected:
        without_penalty = represent_penalty(without_item, without_metrics)
        for with_item in with_selected:
            with_penalty = represent_penalty(with_item, with_metrics)
            gap = without_item.candidate.target_amp - with_item.candidate.target_amp
            total_penalty = without_penalty + with_penalty + 0.35 * abs(gap - gap_target) / gap_scale
            row = (total_penalty, gap, without_item, with_item)
            if best_row is None:
                best_row = row
                continue
            if row[0] < best_row[0] - 1e-9:
                best_row = row
                continue
            if math.isclose(row[0], best_row[0], rel_tol=0.0, abs_tol=1e-9) and row[1] > best_row[1]:
                best_row = row

    if best_row is None:
        raise ValueError("Unable to choose a representative pair.")
    return best_row[2], best_row[3]


def shift_periodic_segment(segment: np.ndarray, sample_rate: float, shift_seconds: float) -> np.ndarray:
    if segment.size <= 1 or np.isclose(shift_seconds, 0.0):
        return segment.copy()
    time_s = np.arange(segment.size, dtype=np.float64) / sample_rate
    duration_s = segment.size / sample_rate
    query = time_s + shift_seconds
    extended_time = np.concatenate([time_s - duration_s, time_s, time_s + duration_s])
    extended_signal = np.tile(segment, 3)
    return np.interp(query, extended_time, extended_signal)


def align_selected_segment(
    state_key: str,
    filtered_signal: np.ndarray,
    selected: SelectedWindow,
    sample_rate: float,
    target_freq: float,
    x_axis_mode: str,
) -> AlignedSegment:
    candidate = selected.candidate
    segment = filtered_signal[candidate.start_index:candidate.end_index]
    time_s = np.arange(segment.size, dtype=np.float64) / sample_rate
    segment_samples = int(segment.size)
    target_amp, phase_rad = fit_target_component(segment, sample_rate, target_freq)

    if state_key == "no_audio" or np.isclose(target_amp, 0.0):
        phase_shift_s = 0.0
        aligned_segment = segment.copy()
    else:
        center_time_s = 0.5 * (time_s[0] + time_s[-1]) if time_s.size else 0.0
        phase_at_center = wrap_to_pi((2.0 * np.pi * target_freq * center_time_s) + phase_rad)
        phase_shift_s = -phase_at_center / (2.0 * np.pi * target_freq)
        aligned_segment = shift_periodic_segment(segment, sample_rate, phase_shift_s)

    if x_axis_mode == "cycles":
        x_axis = time_s * target_freq
    else:
        x_axis = time_s

    return AlignedSegment(
        state_key=state_key,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        center_s=candidate.center_s,
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        target_amp=target_amp,
        phase_rad=phase_rad,
        phase_shift_s=phase_shift_s,
        time_s=time_s,
        x_axis=x_axis,
        raw_segment=segment,
        aligned_segment=aligned_segment,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata_csv(output_csv: Path, segments: list[AlignedSegment]) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "state",
                "sample_rate_hz",
                "selected_start_s",
                "selected_end_s",
                "selected_center_s",
                "target_amp",
                "phase_rad",
                "phase_shift_s",
                "segment_samples",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    segment.state_key,
                    f"{segment.sample_rate:.6f}",
                    f"{segment.start_s:.6f}",
                    f"{segment.end_s:.6f}",
                    f"{segment.center_s:.6f}",
                    f"{segment.target_amp:.12f}",
                    f"{segment.phase_rad:.12f}",
                    f"{segment.phase_shift_s:.12f}",
                    segment.segment_samples,
                ]
            )


def plot_segments(
    png_path: Path,
    svg_path: Path,
    plot_spec: PlotSpec,
    segments: list[AlignedSegment],
    freq_label: str,
    target_freq: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for segment in segments:
        style = SERIES_STYLE[segment.state_key]
        ax.plot(
            segment.x_axis,
            segment.aligned_segment,
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    if plot_spec.duration_mode == "cycles":
        x_max = plot_spec.duration_value
    else:
        x_max = compute_window_seconds(plot_spec, target_freq)
    title = plot_spec.title_template.format(freq_label=freq_label, duration=compute_window_seconds(plot_spec, target_freq), cycles=plot_spec.duration_value)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel(plot_spec.x_label)
    ax.set_ylabel("ADC")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(svg_path)
    plt.close(fig)


def load_filtered_signals(repeat_dir: Path, sample_rate: float, target_freq: float, band_half_width: float, transition_hz: float) -> dict[str, np.ndarray]:
    band_low = max(0.0, target_freq - band_half_width)
    band_high = target_freq + band_half_width
    filtered: dict[str, np.ndarray] = {}
    raw_dir = repeat_dir / "raw"
    for state_key, filename in RAW_SIGNAL_FILES.items():
        centered = read_single_column_signal(raw_dir / filename)
        filtered[state_key] = apply_bandpass_filter(centered, sample_rate, band_low, band_high, transition_hz)
    return filtered


def process_plot_spec(repeat_dir: Path, output_root: Path, plot_spec: PlotSpec, sample_rate: float, target_freq: float, band_half_width: float, transition_hz: float, settle_seconds: float, top_k: int, dpi: int) -> None:
    filtered = load_filtered_signals(repeat_dir, sample_rate, target_freq, band_half_width, transition_hz)
    segment_seconds = compute_window_seconds(plot_spec, target_freq)
    segment_samples = max(2, int(round(segment_seconds * sample_rate)))

    no_audio_candidates = build_candidate_windows(filtered["no_audio"], sample_rate, target_freq, segment_seconds, settle_seconds)
    without_candidates = build_candidate_windows(filtered["without_object"], sample_rate, target_freq, segment_seconds, settle_seconds)
    with_candidates = build_candidate_windows(filtered["with_object"], sample_rate, target_freq, segment_seconds, settle_seconds)

    if not no_audio_candidates or not without_candidates or not with_candidates:
        raise ValueError(f"Unable to build candidates for {repeat_dir.parent.name}/{repeat_dir.name}")

    no_audio_selected = select_quiet_windows(no_audio_candidates, top_k, segment_samples)
    without_selected = select_signal_windows("without_object", without_candidates, top_k, segment_samples)
    with_selected = select_signal_windows("with_object", with_candidates, top_k, segment_samples)

    no_audio_choice = choose_representative_window(no_audio_selected, aggregate_selected_windows(no_audio_selected))
    without_choice, with_choice = choose_representative_pair(without_selected, with_selected)

    segments = [
        align_selected_segment("no_audio", filtered["no_audio"], no_audio_choice, sample_rate, target_freq, plot_spec.duration_mode),
        align_selected_segment("without_object", filtered["without_object"], without_choice, sample_rate, target_freq, plot_spec.duration_mode),
        align_selected_segment("with_object", filtered["with_object"], with_choice, sample_rate, target_freq, plot_spec.duration_mode),
    ]

    frequency_name = repeat_dir.parent.name
    repeat_name = repeat_dir.name
    base_dir = ensure_dir(output_root / frequency_name / plot_spec.output_name)
    png_dir = ensure_dir(base_dir / "png")
    svg_dir = ensure_dir(base_dir / "svg")
    csv_dir = ensure_dir(base_dir / "csv")

    freq_label = format_frequency_label(target_freq)
    output_stem = f"{repeat_name}_{plot_spec.output_name}"
    plot_segments(
        png_dir / f"{output_stem}.png",
        svg_dir / f"{output_stem}.svg",
        plot_spec,
        segments,
        freq_label,
        target_freq,
        dpi,
    )
    write_metadata_csv(csv_dir / f"{output_stem}.csv", segments)


def iter_frequency_dirs(root: Path, only_names: set[str] | None) -> list[Path]:
    dirs = [path for path in root.iterdir() if path.is_dir() and FREQUENCY_PATTERN.match(path.name)]
    if only_names is not None:
        dirs = [path for path in dirs if path.name in only_names]
    return sorted(dirs, key=frequency_sort_key)


def iter_repeat_dirs(frequency_dir: Path, only_names: set[str] | None) -> list[Path]:
    dirs = [path for path in frequency_dir.iterdir() if path.is_dir() and REPEAT_PATTERN.match(path.name)]
    if only_names is not None:
        dirs = [path for path in dirs if path.name in only_names]
    return sorted(dirs, key=repeat_sort_key)


def main() -> None:
    args = parse_args()
    frequency_filter = set(args.frequencies) if args.frequencies else None
    repeat_filter = set(args.repeats) if args.repeats else None

    processed = 0
    for frequency_dir in iter_frequency_dirs(args.frequency_root, frequency_filter):
        target_freq = infer_target_freq(frequency_dir.name)
        sample_rate = target_freq * args.sample_rate_multiplier
        for repeat_dir in iter_repeat_dirs(frequency_dir, repeat_filter):
            for plot_spec in PLOT_SPECS:
                process_plot_spec(
                    repeat_dir,
                    args.output_root,
                    plot_spec,
                    sample_rate,
                    target_freq,
                    args.band_half_width,
                    args.transition_hz,
                    args.settle_seconds,
                    args.top_k,
                    args.dpi,
                )
            processed += 1
            print(f"Generated phase-aligned plots for {frequency_dir.name}/{repeat_dir.name}")

    print(f"Complete. Generated phase-aligned outputs for {processed} repeats.")


if __name__ == "__main__":
    main()

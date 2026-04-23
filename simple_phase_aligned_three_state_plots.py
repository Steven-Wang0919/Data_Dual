from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FREQUENCY_ROOT = SCRIPT_DIR / "frequency_sweep"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "Out_Pic_PhaseAligned_Simple"
FREQUENCY_PATTERN = re.compile(r"^(\d+)\s*Hz$", flags=re.IGNORECASE)
REPEAT_PATTERN = re.compile(r"^repeat_(\d+)$", flags=re.IGNORECASE)
RAW_SIGNAL_FILES = {
    "no_audio": "no_audio.csv",
    "without_object": "without_object.csv",
    "with_object": "with_object.csv",
}
PLOT_NAME = "three_state_phase_aligned_waveform"
FIGSIZE = (12, 4)
LEGEND_STYLE = {
    "frameon": True,
    "framealpha": 0.92,
    "facecolor": "white",
}
SERIES_STYLE = {
    "no_audio": {
        "label": "no audio (reference)",
        "color": "#9ca3af",
        "linewidth": 1.0,
        "alpha": 0.72,
    },
    "without_object": {
        "label": "without object",
        "color": "#2563eb",
        "linewidth": 1.2,
        "alpha": 1.0,
    },
    "with_object": {
        "label": "with object",
        "color": "#dc2626",
        "linewidth": 1.2,
        "alpha": 1.0,
    },
}


@dataclass(frozen=True)
class CandidateWindow:
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    score: float
    rms: float
    envelope_cv: float
    center_distance_s: float


@dataclass(frozen=True)
class SignalBundle:
    centered: np.ndarray
    filtered: np.ndarray


@dataclass(frozen=True)
class AlignedSegment:
    state_key: str
    sample_rate: float
    start_s: float
    end_s: float
    phase_shift_s: float
    phase_rad: float
    segment_samples: int
    time_axis: np.ndarray
    original_segment: np.ndarray
    aligned_segment: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render simple phase-aligned three-state waveform plots without modifying frequency_sweep."
    )
    parser.add_argument(
        "--frequency-root",
        type=Path,
        default=DEFAULT_FREQUENCY_ROOT,
        help="Root directory that contains 0100Hz/repeat_01 style experiment folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for simple phase-aligned plot exports.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        default=None,
        help="Frequency directories to process. Defaults to all discovered frequencies.",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Repeat directories to process. Defaults to all discovered repeats under each frequency.",
    )
    parser.add_argument(
        "--representative-seconds",
        type=float,
        default=0.2,
        help="Length of the phase-aligned short window.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Ignore the first and last part of each record when selecting candidate windows.",
    )
    parser.add_argument(
        "--band-half-width",
        type=float,
        default=10.0,
        help="Half-width of the band-pass filter around the target frequency.",
    )
    parser.add_argument(
        "--transition-hz",
        type=float,
        default=5.0,
        help="Transition width used by the cosine-tapered FFT band-pass filter.",
    )
    parser.add_argument(
        "--sample-rate-multiplier",
        type=float,
        default=10.0,
        help="Nominal sample-rate rule applied as sample_rate = target_freq * multiplier.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def frequency_sort_key(name: str) -> float:
    return infer_target_freq(name)


def discover_frequencies(frequency_root: Path) -> list[str]:
    return sorted(
        [
            entry.name
            for entry in frequency_root.iterdir()
            if entry.is_dir() and FREQUENCY_PATTERN.match(entry.name)
        ],
        key=frequency_sort_key,
    )


def discover_repeats(frequency_dir: Path) -> list[str]:
    def repeat_key(name: str) -> int:
        match = REPEAT_PATTERN.match(name)
        return int(match.group(1)) if match else sys.maxsize

    repeats = [
        entry.name
        for entry in frequency_dir.iterdir()
        if entry.is_dir() and REPEAT_PATTERN.match(entry.name)
    ]
    return sorted(repeats, key=repeat_key)


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


def read_single_column_signal(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            values.append(float(row[0]))

    if not values:
        raise ValueError(f"No signal samples found in {path}")

    signal = np.asarray(values, dtype=np.float64)
    return signal - np.mean(signal)


def build_bandpass_response(
    freqs: np.ndarray,
    band_low: float,
    band_high: float,
    transition_hz: float,
) -> np.ndarray:
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


def apply_bandpass_filter(
    centered_signal: np.ndarray,
    sample_rate: float,
    band_low: float,
    band_high: float,
    transition_hz: float,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(centered_signal.size, d=1.0 / sample_rate)
    response = build_bandpass_response(freqs, band_low, band_high, transition_hz)
    filtered_fft = np.fft.rfft(centered_signal) * response
    return np.fft.irfft(filtered_fft, n=centered_signal.size)


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    sample_count = signal.size
    if sample_count == 0:
        return np.asarray([], dtype=np.float64)

    spectrum = np.fft.fft(signal)
    hilbert_multiplier = np.zeros(sample_count, dtype=np.float64)
    if sample_count % 2 == 0:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[sample_count // 2] = 1.0
        hilbert_multiplier[1 : sample_count // 2] = 2.0
    else:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[1 : (sample_count + 1) // 2] = 2.0

    analytic_signal = np.fft.ifft(spectrum * hilbert_multiplier)
    return np.abs(analytic_signal)


def build_candidate_windows(
    filtered_signal: np.ndarray,
    sample_rate: float,
    target_freq: float,
    representative_seconds: float,
    settle_seconds: float,
) -> list[CandidateWindow]:
    total_samples = filtered_signal.size
    if total_samples == 0:
        return []

    segment_samples = max(2, int(round(representative_seconds * sample_rate)))
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
        rms = float(np.sqrt(np.mean(np.square(segment))))
        envelope = analytic_envelope(segment)
        envelope_mean = max(float(np.mean(envelope)), 1e-12)
        envelope_cv = float(np.std(envelope) / envelope_mean)
        score = rms / (1.0 + envelope_cv)
        start_s = start_index / sample_rate
        end_s = end_index / sample_rate
        center_s = 0.5 * (start_s + end_s)
        candidates.append(
            CandidateWindow(
                start_index=start_index,
                end_index=end_index,
                start_s=start_s,
                end_s=end_s,
                score=score,
                rms=rms,
                envelope_cv=envelope_cv,
                center_distance_s=abs(center_s - total_center_s),
            )
        )

    return candidates


def select_candidate(state_key: str, candidates: list[CandidateWindow]) -> CandidateWindow:
    if not candidates:
        raise ValueError("Cannot select a candidate window from an empty candidate list.")

    if state_key == "no_audio":
        return min(candidates, key=lambda candidate: (candidate.rms, candidate.center_distance_s))

    return max(
        candidates,
        key=lambda candidate: (candidate.score, candidate.rms, -candidate.center_distance_s),
    )


def fit_phase_at_target(
    segment: np.ndarray,
    sample_rate: float,
    target_freq: float,
) -> tuple[float, float]:
    local_time = np.arange(segment.size, dtype=np.float64) / sample_rate
    center_time = 0.5 * (segment.size / sample_rate)
    centered_time = local_time - center_time
    angular_freq = 2.0 * np.pi * target_freq
    design = np.column_stack(
        (
            np.sin(angular_freq * centered_time),
            np.cos(angular_freq * centered_time),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, segment, rcond=None)
    sine_coeff, cosine_coeff = coefficients
    amplitude = float(np.hypot(sine_coeff, cosine_coeff))
    phase_rad = float(np.arctan2(sine_coeff, cosine_coeff))
    return amplitude, phase_rad


def interpolate_with_linear_edges(
    sample_times: np.ndarray,
    signal: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    result = np.interp(query_times, sample_times, signal)
    if signal.size >= 2:
        left_mask = query_times < sample_times[0]
        if np.any(left_mask):
            left_slope = (signal[1] - signal[0]) / (sample_times[1] - sample_times[0])
            result[left_mask] = signal[0] + left_slope * (query_times[left_mask] - sample_times[0])

        right_mask = query_times > sample_times[-1]
        if np.any(right_mask):
            right_slope = (signal[-1] - signal[-2]) / (sample_times[-1] - sample_times[-2])
            result[right_mask] = signal[-1] + right_slope * (query_times[right_mask] - sample_times[-1])
    return result


def align_segment_phase(
    state_key: str,
    signal: np.ndarray,
    sample_rate: float,
    target_freq: float,
    candidate: CandidateWindow,
) -> AlignedSegment:
    segment = signal[candidate.start_index : candidate.end_index]
    amplitude, phase_rad = fit_phase_at_target(segment, sample_rate, target_freq)
    segment_rms = float(np.sqrt(np.mean(np.square(segment))))

    phase_shift_s = phase_rad / (2.0 * np.pi * target_freq)
    if state_key == "no_audio" and amplitude < max(1e-6, 0.05 * segment_rms):
        phase_shift_s = 0.0
        phase_rad = 0.0

    time_axis = np.arange(segment.size, dtype=np.float64) / sample_rate
    aligned_segment = interpolate_with_linear_edges(
        time_axis,
        segment,
        time_axis + phase_shift_s,
    )

    return AlignedSegment(
        state_key=state_key,
        sample_rate=sample_rate,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        phase_shift_s=phase_shift_s,
        phase_rad=phase_rad,
        segment_samples=segment.size,
        time_axis=time_axis,
        original_segment=segment,
        aligned_segment=aligned_segment,
    )


def style_legend(ax: plt.Axes) -> None:
    legend = ax.legend(loc="lower right", **LEGEND_STYLE)
    if legend is not None:
        legend.get_frame().set_edgecolor("#d1d5db")


def save_plot(
    output_root: Path,
    frequency_name: str,
    repeat_name: str,
    aligned_segments: list[AlignedSegment],
    target_freq: float,
    dpi: int,
) -> None:
    plot_root = output_root / frequency_name / PLOT_NAME
    png_dir = plot_root / "png"
    svg_dir = plot_root / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=FIGSIZE)
    for aligned_segment in aligned_segments:
        style = SERIES_STYLE[aligned_segment.state_key]
        axis.plot(
            aligned_segment.time_axis,
            aligned_segment.aligned_segment,
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    duration_s = aligned_segments[0].segment_samples / aligned_segments[0].sample_rate
    freq_label = format_frequency_label(target_freq)
    axis.set_xlim(0.0, duration_s)
    axis.set_xlabel("Time Within Phase-Aligned Window (s)")
    axis.set_ylabel("ADC")
    axis.set_title(f"Phase-Aligned {freq_label} Waveform")
    style_legend(axis)
    figure.tight_layout()

    base_name = f"{repeat_name}_{PLOT_NAME}"
    figure.savefig(png_dir / f"{base_name}.png", dpi=dpi)
    figure.savefig(svg_dir / f"{base_name}.svg")
    plt.close(figure)


def write_sidecar_csv(
    output_root: Path,
    frequency_name: str,
    repeat_name: str,
    aligned_segments: list[AlignedSegment],
) -> None:
    csv_dir = output_root / frequency_name / PLOT_NAME / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{repeat_name}_{PLOT_NAME}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "state",
                "sample_rate",
                "selected_start_s",
                "selected_end_s",
                "phase_shift_s",
                "phase_rad",
                "segment_samples",
            )
        )
        for aligned_segment in aligned_segments:
            writer.writerow(
                (
                    aligned_segment.state_key,
                    f"{aligned_segment.sample_rate:.6f}",
                    f"{aligned_segment.start_s:.6f}",
                    f"{aligned_segment.end_s:.6f}",
                    f"{aligned_segment.phase_shift_s:.9f}",
                    f"{aligned_segment.phase_rad:.9f}",
                    aligned_segment.segment_samples,
                )
            )


def load_signal_bundles(
    repeat_dir: Path,
    sample_rate: float,
    target_freq: float,
    band_half_width: float,
    transition_hz: float,
) -> dict[str, SignalBundle]:
    raw_dir = repeat_dir / "raw"
    band_low = max(0.0, target_freq - band_half_width)
    band_high = target_freq + band_half_width
    bundles: dict[str, SignalBundle] = {}
    for state_key, filename in RAW_SIGNAL_FILES.items():
        centered = read_single_column_signal(raw_dir / filename)
        filtered = apply_bandpass_filter(
            centered,
            sample_rate,
            band_low,
            band_high,
            transition_hz,
        )
        bundles[state_key] = SignalBundle(centered=centered, filtered=filtered)
    return bundles


def process_repeat(
    repeat_dir: Path,
    output_root: Path,
    representative_seconds: float,
    settle_seconds: float,
    band_half_width: float,
    transition_hz: float,
    sample_rate_multiplier: float,
    dpi: int,
) -> bool:
    frequency_name = repeat_dir.parent.name
    target_freq = infer_target_freq(frequency_name)
    sample_rate = target_freq * sample_rate_multiplier

    raw_dir = repeat_dir / "raw"
    missing_raw = [
        filename
        for filename in RAW_SIGNAL_FILES.values()
        if not (raw_dir / filename).is_file()
    ]
    if missing_raw:
        warn(
            f"Skipping {frequency_name}/{repeat_dir.name} because raw files are missing: "
            f"{', '.join(missing_raw)}"
        )
        return False

    try:
        signal_bundles = load_signal_bundles(
            repeat_dir,
            sample_rate,
            target_freq,
            band_half_width,
            transition_hz,
        )
    except (OSError, ValueError) as exc:
        warn(f"Skipping {frequency_name}/{repeat_dir.name}: {exc}")
        return False

    aligned_segments: list[AlignedSegment] = []
    for state_key in ("no_audio", "without_object", "with_object"):
        candidates = build_candidate_windows(
            signal_bundles[state_key].filtered,
            sample_rate,
            target_freq,
            representative_seconds,
            settle_seconds,
        )
        if not candidates:
            warn(f"Skipping {frequency_name}/{repeat_dir.name}: no candidate windows for {state_key}.")
            return False
        selected_candidate = select_candidate(state_key, candidates)
        aligned_segments.append(
            align_segment_phase(
                state_key,
                signal_bundles[state_key].filtered,
                sample_rate,
                target_freq,
                selected_candidate,
            )
        )

    save_plot(output_root, frequency_name, repeat_dir.name, aligned_segments, target_freq, dpi)
    write_sidecar_csv(output_root, frequency_name, repeat_dir.name, aligned_segments)
    print(f"Generated {frequency_name}/{repeat_dir.name} -> {output_root / frequency_name / PLOT_NAME}")
    return True


def main() -> None:
    args = parse_args()
    frequency_root = args.frequency_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not frequency_root.is_dir():
        raise FileNotFoundError(f"Frequency root not found: {frequency_root}")

    frequencies = args.frequencies or discover_frequencies(frequency_root)
    if not frequencies:
        raise FileNotFoundError("No frequency directories were discovered.")

    processed_count = 0
    for frequency_name in frequencies:
        frequency_dir = frequency_root / frequency_name
        if not frequency_dir.is_dir():
            warn(f"Skipping missing frequency directory: {frequency_dir}")
            continue

        repeats = args.repeats or discover_repeats(frequency_dir)
        if not repeats:
            warn(f"Skipping {frequency_name} because no repeat directories were discovered.")
            continue

        for repeat_name in repeats:
            repeat_dir = frequency_dir / repeat_name
            if not repeat_dir.is_dir():
                warn(f"Skipping missing repeat directory: {repeat_dir}")
                continue
            if process_repeat(
                repeat_dir,
                output_root,
                args.representative_seconds,
                args.settle_seconds,
                args.band_half_width,
                args.transition_hz,
                args.sample_rate_multiplier,
                args.dpi,
            ):
                processed_count += 1

    if processed_count == 0:
        raise RuntimeError("No phase-aligned plots were generated.")

    print(f"Simple phase-aligned three-state plot generation complete. Generated {processed_count} repeat(s).")


if __name__ == "__main__":
    main()

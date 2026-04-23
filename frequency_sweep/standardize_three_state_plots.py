from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DUAL_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_PIC_ROOT = DATA_DUAL_DIR / "Out_Pic"
FREQUENCY_PATTERN = re.compile(r"^(\d+)\s*Hz$", flags=re.IGNORECASE)
ANALYSIS_SUBDIR = Path("analysis") / "three_state_analysis"
RAW_SIGNAL_FILES = {
    "no_audio": "no_audio.csv",
    "without_object": "without_object.csv",
    "with_object": "with_object.csv",
}
FIGSIZE = (12, 4)
DEFAULT_DPI = 150
LEGEND_STYLE = {
    "frameon": True,
    "framealpha": 0.92,
    "facecolor": "white",
}


@dataclass(frozen=True)
class WindowSelection:
    start_s: float
    end_s: float
    sample_count: int


@dataclass(frozen=True)
class LineSeriesSpec:
    state_key: str
    label: str
    color: str


@dataclass(frozen=True)
class WaveformPlotSpec:
    output_name: str
    window_csv: str
    legend_loc: str
    title_template: str
    linewidth: float
    series: tuple[LineSeriesSpec, ...]


@dataclass(frozen=True)
class SpectrumPlotSpec:
    output_name: str
    title: str
    legend_loc: str
    linewidth: float
    series: tuple[LineSeriesSpec, ...]


@dataclass(frozen=True)
class SignalBundle:
    centered: np.ndarray
    filtered: np.ndarray
    freqs: np.ndarray
    raw_amplitudes: np.ndarray


WAVEFORM_PLOT_SPECS = (
    WaveformPlotSpec(
        output_name="three_state_filtered_waveform_detail",
        window_csv="three_state_detail_windows.csv",
        legend_loc="upper left",
        title_template="{freq_label} Band-Pass Detail",
        linewidth=1.1,
        series=(
            LineSeriesSpec("no_audio", "no audio", "#6b7280"),
            LineSeriesSpec("without_object", "without object", "#2563eb"),
            LineSeriesSpec("with_object", "with object", "#dc2626"),
        ),
    ),
    WaveformPlotSpec(
        output_name="three_state_representative_filtered_waveform",
        window_csv="three_state_representative_windows.csv",
        legend_loc="lower right",
        title_template="Representative {freq_label} Waveform",
        linewidth=1.1,
        series=(
            LineSeriesSpec("no_audio", "no audio (reference)", "#9ca3af"),
            LineSeriesSpec("without_object", "without object", "#2563eb"),
            LineSeriesSpec("with_object", "with object", "#dc2626"),
        ),
    ),
)

SPECTRUM_PLOT_SPEC = SpectrumPlotSpec(
    output_name="three_state_spectrum",
    title="Object-Focused Spectrum Comparison",
    legend_loc="upper right",
    linewidth=0.9,
    series=(
        LineSeriesSpec("no_audio", "no audio (reference)", "#9ca3af"),
        LineSeriesSpec("without_object", "without object", "#2563eb"),
        LineSeriesSpec("with_object", "with object", "#dc2626"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render standardized three-state plots and sync them into Out_Pic."
    )
    parser.add_argument(
        "--frequency-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Root directory that contains 0100Hz/repeat_01 style experiment folders.",
    )
    parser.add_argument(
        "--out-pic-root",
        type=Path,
        default=DEFAULT_OUT_PIC_ROOT,
        help="Output root for synchronized plots.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        default=None,
        help="Frequency directories to process. Defaults to the frequencies already present in Out_Pic.",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=[f"repeat_{index:02d}" for index in range(1, 6)],
        help="Repeat directories to process.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Optional sample-rate override. When omitted the script derives it from three_state_waveforms.csv.",
    )
    parser.add_argument(
        "--band-half-width",
        type=float,
        default=10.0,
        help="Half-width of the highlighted band around the target frequency.",
    )
    parser.add_argument(
        "--transition-hz",
        type=float,
        default=5.0,
        help="Transition width used by the cosine-tapered FFT band-pass filter.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def discover_frequencies(frequency_root: Path, out_pic_root: Path) -> list[str]:
    if out_pic_root.is_dir():
        candidates = [
            entry.name
            for entry in out_pic_root.iterdir()
            if entry.is_dir() and FREQUENCY_PATTERN.match(entry.name)
        ]
        if candidates:
            return sorted(candidates, key=frequency_sort_key)

    candidates = [
        entry.name
        for entry in frequency_root.iterdir()
        if entry.is_dir() and FREQUENCY_PATTERN.match(entry.name)
    ]
    return sorted(candidates, key=frequency_sort_key)


def frequency_sort_key(name: str) -> float:
    return infer_target_freq(name)


def infer_target_freq(frequency_name: str) -> float:
    match = FREQUENCY_PATTERN.search(frequency_name)
    if match is None:
        raise ValueError(f"Unable to infer target frequency from directory name: {frequency_name}")
    return float(match.group(1))


def format_frequency_label(target_freq: float) -> str:
    rounded = round(target_freq)
    if np.isclose(target_freq, rounded):
        return f"{int(rounded)}Hz"
    return f"{target_freq:g}Hz"


def read_sample_rate(analysis_dir: Path, sample_rate_override: float | None) -> float:
    if sample_rate_override is not None:
        return float(sample_rate_override)

    waveform_csv = analysis_dir / "three_state_waveforms.csv"
    with waveform_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        times: list[float] = []
        for row in reader:
            times.append(float(row["time_s"]))
            if len(times) >= 8:
                break

    if len(times) < 2:
        raise ValueError(f"Unable to derive sample rate from {waveform_csv}")

    diffs = np.diff(np.asarray(times, dtype=np.float64))
    step = float(np.median(diffs))
    if step <= 0.0:
        raise ValueError(f"Invalid time step derived from {waveform_csv}: {step}")
    return 1.0 / step


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
    return np.asarray(values, dtype=np.float64)


def load_window_selections(path: Path) -> dict[str, WindowSelection]:
    selections: dict[str, WindowSelection] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["label"].strip()
            selections[label] = WindowSelection(
                start_s=float(row["selected_start_s"]),
                end_s=float(row["selected_end_s"]),
                sample_count=int(row["segment_samples"]),
            )

    expected = set(RAW_SIGNAL_FILES)
    missing = expected.difference(selections)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing window selections in {path}: {missing_text}")
    return selections


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


def compute_amplitude_spectrum(centered_signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(centered_signal)
    amplitudes = np.abs(spectrum) / centered_signal.size
    if amplitudes.size > 1:
        if centered_signal.size % 2 == 0 and amplitudes.size > 2:
            amplitudes[1:-1] *= 2.0
        elif centered_signal.size % 2 != 0:
            amplitudes[1:] *= 2.0
    freqs = np.fft.rfftfreq(centered_signal.size, d=1.0 / sample_rate)
    return freqs, amplitudes


def load_signal_bundles(
    repeat_dir: Path,
    sample_rate: float,
    target_freq: float,
    band_half_width: float,
    transition_hz: float,
) -> dict[str, SignalBundle]:
    band_low = max(0.0, target_freq - band_half_width)
    band_high = target_freq + band_half_width
    raw_dir = repeat_dir / "raw"
    bundles: dict[str, SignalBundle] = {}

    for state_key, raw_filename in RAW_SIGNAL_FILES.items():
        raw_signal = read_single_column_signal(raw_dir / raw_filename)
        centered_signal = raw_signal - np.mean(raw_signal)
        filtered_signal = apply_bandpass_filter(
            centered_signal,
            sample_rate,
            band_low,
            band_high,
            transition_hz,
        )
        freqs, raw_amplitudes = compute_amplitude_spectrum(centered_signal, sample_rate)
        bundles[state_key] = SignalBundle(
            centered=centered_signal,
            filtered=filtered_signal,
            freqs=freqs,
            raw_amplitudes=raw_amplitudes,
        )

    return bundles


def extract_segment(
    signal: np.ndarray,
    selection: WindowSelection,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    expected_duration = selection.sample_count / sample_rate
    actual_duration = selection.end_s - selection.start_s
    if not np.isclose(actual_duration, expected_duration, atol=max(1e-9, 1.0 / sample_rate)):
        raise ValueError(
            "Window duration does not match segment_samples: "
            f"start={selection.start_s}, end={selection.end_s}, samples={selection.sample_count}"
        )

    start_index = int(round(selection.start_s * sample_rate))
    end_index = start_index + selection.sample_count
    if start_index < 0 or end_index > signal.size:
        raise ValueError(
            "Selected window falls outside the available raw signal: "
            f"start_index={start_index}, end_index={end_index}, signal_size={signal.size}"
        )

    local_time = np.arange(selection.sample_count, dtype=np.float64) / sample_rate
    return local_time, signal[start_index:end_index]


def style_legend(ax: plt.Axes, legend_loc: str) -> None:
    legend = ax.legend(loc=legend_loc, **LEGEND_STYLE)
    if legend is None:
        return
    legend.get_frame().set_edgecolor("#d1d5db")


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def render_waveform_plot(
    analysis_dir: Path,
    plot_spec: WaveformPlotSpec,
    signal_bundles: dict[str, SignalBundle],
    sample_rate: float,
    freq_label: str,
    dpi: int,
) -> None:
    selections = load_window_selections(analysis_dir / plot_spec.window_csv)
    durations = [selection.sample_count / sample_rate for selection in selections.values()]
    duration_s = max(durations)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for series in plot_spec.series:
        selection = selections[series.state_key]
        local_time, segment = extract_segment(
            signal_bundles[series.state_key].filtered,
            selection,
            sample_rate,
        )
        ax.plot(
            local_time,
            segment,
            label=series.label,
            linewidth=plot_spec.linewidth,
            color=series.color,
        )

    title = plot_spec.title_template.format(freq_label=freq_label, duration=duration_s)
    ax.set_xlim(0.0, duration_s)
    ax.set_xlabel("Time Within Selected Window (s)")
    ax.set_ylabel("ADC")
    ax.set_title(title)
    style_legend(ax, plot_spec.legend_loc)
    save_figure(fig, analysis_dir / plot_spec.output_name, dpi)


def render_spectrum_plot(
    analysis_dir: Path,
    signal_bundles: dict[str, SignalBundle],
    target_freq: float,
    band_half_width: float,
    dpi: int,
) -> None:
    spectrum_limit = min(
        500.0,
        *(
            float(bundle.freqs[-1])
            for bundle in signal_bundles.values()
            if bundle.freqs.size
        ),
    )
    target_focus_low = max(0.0, target_freq - max(40.0, target_freq * 0.4))
    target_focus_high = min(spectrum_limit, target_freq + max(120.0, target_freq * 1.5))
    band_low = max(0.0, target_freq - band_half_width)
    band_high = target_freq + band_half_width

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for series in SPECTRUM_PLOT_SPEC.series:
        bundle = signal_bundles[series.state_key]
        limit_index = np.searchsorted(bundle.freqs, spectrum_limit, side="right")
        ax.plot(
            bundle.freqs[:limit_index],
            bundle.raw_amplitudes[:limit_index],
            label=series.label,
            linewidth=SPECTRUM_PLOT_SPEC.linewidth,
            color=series.color,
        )

    ax.axvspan(band_low, band_high, color="#fde68a", alpha=0.35)
    ax.axvline(target_freq, color="#7c3aed", linestyle="--", linewidth=1.0)
    ax.set_xlim(target_focus_low, target_focus_high)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(SPECTRUM_PLOT_SPEC.title)
    style_legend(ax, SPECTRUM_PLOT_SPEC.legend_loc)
    save_figure(fig, analysis_dir / SPECTRUM_PLOT_SPEC.output_name, dpi)


def sync_plot_to_out_pic(
    analysis_dir: Path,
    out_pic_root: Path,
    frequency_name: str,
    repeat_name: str,
    plot_name: str,
) -> None:
    for extension in ("png", "svg"):
        source_path = analysis_dir / f"{plot_name}.{extension}"
        target_dir = out_pic_root / frequency_name / plot_name / extension
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{repeat_name}_{plot_name}.{extension}"
        shutil.copyfile(source_path, target_path)


def process_repeat(
    repeat_dir: Path,
    out_pic_root: Path,
    sample_rate_override: float | None,
    band_half_width: float,
    transition_hz: float,
    dpi: int,
) -> None:
    analysis_dir = repeat_dir / ANALYSIS_SUBDIR
    frequency_name = repeat_dir.parent.name
    target_freq = infer_target_freq(frequency_name)
    freq_label = format_frequency_label(target_freq)
    sample_rate = read_sample_rate(analysis_dir, sample_rate_override)
    signal_bundles = load_signal_bundles(
        repeat_dir,
        sample_rate,
        target_freq,
        band_half_width,
        transition_hz,
    )

    for plot_spec in WAVEFORM_PLOT_SPECS:
        render_waveform_plot(
            analysis_dir,
            plot_spec,
            signal_bundles,
            sample_rate,
            freq_label,
            dpi,
        )
        sync_plot_to_out_pic(
            analysis_dir,
            out_pic_root,
            frequency_name,
            repeat_dir.name,
            plot_spec.output_name,
        )

    render_spectrum_plot(
        analysis_dir,
        signal_bundles,
        target_freq,
        band_half_width,
        dpi,
    )
    sync_plot_to_out_pic(
        analysis_dir,
        out_pic_root,
        frequency_name,
        repeat_dir.name,
        SPECTRUM_PLOT_SPEC.output_name,
    )

    print(f"Standardized {frequency_name}/{repeat_dir.name}")


def main() -> None:
    args = parse_args()
    frequency_root = args.frequency_root.expanduser().resolve()
    out_pic_root = args.out_pic_root.expanduser().resolve()
    frequencies = args.frequencies or discover_frequencies(frequency_root, out_pic_root)

    if not frequencies:
        raise FileNotFoundError("No frequency directories were discovered for standardization.")

    for frequency_name in frequencies:
        frequency_dir = frequency_root / frequency_name
        if not frequency_dir.is_dir():
            raise FileNotFoundError(f"Frequency directory not found: {frequency_dir}")
        for repeat_name in args.repeats:
            repeat_dir = frequency_dir / repeat_name
            if not repeat_dir.is_dir():
                raise FileNotFoundError(f"Repeat directory not found: {repeat_dir}")
            process_repeat(
                repeat_dir,
                out_pic_root,
                args.sample_rate,
                args.band_half_width,
                args.transition_hz,
                args.dpi,
            )

    print("Three-state plot standardization complete.")


if __name__ == "__main__":
    main()

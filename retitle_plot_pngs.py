from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
FREQUENCY_PATTERN = re.compile(r"^(\d+)\s*Hz$", flags=re.IGNORECASE)


def format_frequency_label(frequency_name: str) -> str:
    match = FREQUENCY_PATTERN.match(frequency_name)
    if match is None:
        raise ValueError(f"Unable to infer frequency from {frequency_name}")
    return f"{int(match.group(1))}Hz"


def load_title_font(image_height: int) -> ImageFont.ImageFont:
    font_size = max(18, int(round(image_height * 0.045)))
    try:
        from matplotlib import font_manager

        font_path = font_manager.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()


def retitle_png(png_path: Path, title: str) -> None:
    with Image.open(png_path) as image:
        canvas = image.convert("RGB")

    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = load_title_font(height)
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    band_bottom = max(54, int(round(height * 0.092)))
    draw.rectangle((0, 0, width, band_bottom), fill="white")

    x = int(round((width - title_width) * 0.5))
    y = max(8, int(round((band_bottom - title_height) * 0.45)))
    draw.text((x, y), title, fill="black", font=font)
    canvas.save(png_path)


def retitle_frequency_sweep_detail() -> int:
    count = 0
    frequency_root = SCRIPT_DIR / "frequency_sweep"
    for frequency_dir in sorted(frequency_root.glob("*Hz")):
        if not frequency_dir.is_dir() or not FREQUENCY_PATTERN.match(frequency_dir.name):
            continue
        title = f"{format_frequency_label(frequency_dir.name)} Band-Pass Detail"
        for png_path in sorted(frequency_dir.glob("repeat_*/analysis/three_state_analysis/three_state_filtered_waveform_detail.png")):
            retitle_png(png_path, title)
            count += 1
    return count


def retitle_organized_detail() -> int:
    count = 0
    organized_root = SCRIPT_DIR / "Out_Pic" / " 4_23(100~1000Hz)"
    for frequency_dir in sorted(organized_root.glob("*Hz")):
        if not frequency_dir.is_dir() or not FREQUENCY_PATTERN.match(frequency_dir.name):
            continue
        title = f"{format_frequency_label(frequency_dir.name)} Band-Pass Detail"
        detail_dir = frequency_dir / "three_state_filtered_waveform_detail"
        for png_path in sorted(detail_dir.glob("*.png")):
            retitle_png(png_path, title)
            count += 1
    return count


def retitle_phase_aligned_outputs() -> int:
    count = 0
    phase_root = SCRIPT_DIR / "Out_Pic_PhaseAligned"
    for frequency_dir in sorted(phase_root.glob("*Hz")):
        if not frequency_dir.is_dir() or not FREQUENCY_PATTERN.match(frequency_dir.name):
            continue
        freq_label = format_frequency_label(frequency_dir.name)
        plot_titles = {
            "three_state_phase_aligned_waveform": f"Phase-Aligned {freq_label} Waveform",
            "three_state_phase_aligned_fixed_cycles_waveform": f"Phase-Aligned {freq_label} Fixed-Cycles Waveform",
        }
        for plot_name, title in plot_titles.items():
            png_dir = frequency_dir / plot_name / "png"
            for png_path in sorted(png_dir.glob("*.png")):
                retitle_png(png_path, title)
                count += 1
    return count


def retitle_organized_fixed_cycles() -> int:
    count = 0
    organized_root = SCRIPT_DIR / "Out_Pic" / " 4_23(100~1000Hz)"
    for frequency_dir in sorted(organized_root.glob("*Hz")):
        if not frequency_dir.is_dir() or not FREQUENCY_PATTERN.match(frequency_dir.name):
            continue
        title = f"Phase-Aligned {format_frequency_label(frequency_dir.name)} Fixed-Cycles Waveform"
        phase_dir = frequency_dir / "three_state_phase_aligned_fixed_cycles_waveform"
        for png_path in sorted(phase_dir.glob("*.png")):
            retitle_png(png_path, title)
            count += 1
    return count


def main() -> None:
    counts = {
        "frequency_sweep_detail": retitle_frequency_sweep_detail(),
        "organized_detail": retitle_organized_detail(),
        "phase_aligned_outputs": retitle_phase_aligned_outputs(),
        "organized_fixed_cycles": retitle_organized_fixed_cycles(),
    }
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()

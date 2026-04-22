# Frequency Sweep Data Layout

This directory stores the new standard dataset layout for the soil foreign-object detection experiment.

## Grouping Logic

- Group by excitation frequency.
- Frequency range: `100 Hz` to `1000 Hz`.
- Step size: `100 Hz`.
- Repeats per frequency: `5`.
- Each repeat keeps three raw states:
  - `no_audio.csv`
  - `without_object.csv`
  - `with_object.csv`

## Directory Structure

```text
frequency_sweep/
|-- experiment_manifest.csv
|-- aggregate_analysis/
|-- 0100Hz/
|   |-- repeat_01/
|   |   |-- raw/
|   |   `-- analysis/
|   |-- repeat_02/
|   |-- repeat_03/
|   |-- repeat_04/
|   `-- repeat_05/
|-- 0200Hz/
|-- 0300Hz/
|-- 0400Hz/
|-- 0500Hz/
|-- 0600Hz/
|-- 0700Hz/
|-- 0800Hz/
|-- 0900Hz/
`-- 1000Hz/
```

## File Naming Recommendation

For each `raw/` directory, use the same file names every time:

- `no_audio.csv`
- `without_object.csv`
- `with_object.csv`

For each `analysis/` directory, recommended output subfolders are:

- `no_audio_analysis`
- `without_object_analysis`
- `with_object_analysis`
- `three_state_analysis`

## Recommended Workflow

1. Put raw CSV files into the corresponding `raw/` folder.
2. Run the single-file analysis script for each raw file when needed.
3. Run the three-state comparison script for each repeat.
4. Run `standardize_three_state_plots.py` to refresh the standardized PNG/SVG exports in both `analysis/three_state_analysis` and `Out_Pic/`.
5. Put cross-frequency summaries into `aggregate_analysis/`.

## Why This Layout Helps

- Frequencies are separated cleanly.
- Repeats are independent and traceable.
- The same scripts can be reused for every frequency.
- Later batch analysis across `100-1000 Hz` is easier.

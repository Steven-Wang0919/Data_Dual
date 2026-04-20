from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repeat_dir = Path(__file__).resolve().parent
    project_root = repeat_dir.parents[3]
    helper_script = project_root / "Data_Dual" / "frequency_sweep" / "run_three_state_repeat.py"
    extra_args = sys.argv[1:]
    if "--sample-rate" not in extra_args:
        extra_args = ["--sample-rate", "6000", *extra_args]
    command = [sys.executable, str(helper_script), str(repeat_dir), *extra_args]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

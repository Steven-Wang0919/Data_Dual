from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repeat_dir = Path(__file__).resolve().parent
    project_root = repeat_dir.parents[3]
    helper_script = project_root / "Data_Dual" / "frequency_sweep" / "run_three_state_repeat.py"
    command = [sys.executable, str(helper_script), str(repeat_dir), *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

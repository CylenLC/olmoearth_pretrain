"""Run the unchanged official recipe with an explicit profiling bootstrap."""

import os
from pathlib import Path
import runpy
import sys

from observer import apply_affinity, install_training

if __name__ == "__main__":
    affinity = apply_affinity("parent")
    entry = Path(__file__).resolve().parents[2] / "scripts/official/v1_2/nano.py"
    sys.path.insert(0, str(entry.parent))
    sys.argv[0] = str(entry)
    export = install_training(
        int(os.environ["OSS_PROFILE_FIRST"]), int(os.environ["OSS_PROFILE_LAST"]), affinity
    )
    runpy.run_path(str(entry), run_name="__main__")
    export()

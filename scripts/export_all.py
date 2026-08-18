"""Run every S0 export in a single process.

Provenance is stamped once per process, so all artifacts from one run share a SHA and a
dirty flag. Running the individual export scripts in sequence would instead let each one
dirty the tree for the next, leaving only the first manifest with a clean stamp -- use
this entry point for the provenance pass of the two-commit pattern (see README).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_s0
import export_differential


def main() -> None:
    print("exporting S0 results...")
    calibrated = export_s0.export_e0_golden()["calibrated"]
    export_s0.export_config_grid()
    export_s0.export_coupling1_ceiling()

    print("exporting differential comparison (n=200000)...")
    export_differential.run()

    import time
    export_s0.write_run_manifest(calibrated, time.perf_counter())
    print("done.")


if __name__ == "__main__":
    main()

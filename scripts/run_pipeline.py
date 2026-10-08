#!/usr/bin/env python
"""End-to-end pipeline runner: raw → normalized → aligned → features → inventory."""

import subprocess
import sys
from pathlib import Path

def run_step(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = repo_root / "scripts"

    # 1. Generate raw synthetic data
    run_step([sys.executable, str(scripts / "generate_sample_raw.py")], cwd=repo_root)

    # 2. Normalize raw data
    run_step([sys.executable, str(scripts / "normalize_raw.py")], cwd=repo_root)

    # 3. Align symbols
    run_step([sys.executable, str(scripts / "align_symbols.py")], cwd=repo_root)

    # 4. Compute features
    run_step([sys.executable, str(scripts / "compute_features.py")], cwd=repo_root)

    # 5. Inventory models
    run_step([sys.executable, str(scripts / "inventory_models.py")], cwd=repo_root)

    print("Pipeline complete. Outputs:")
    print("  - data/raw/          : synthetic 1-min OHLCV")
    print("  - data/normalized/   : calendar-filtered")
    print("  - data/aligned/      : aligned across symbols")
    print("  - data/features/     : computed features")
    print("  - data/manifests/    : model inventory")

if __name__ == "__main__":
    main()
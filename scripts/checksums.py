#!/usr/bin/env python3
"""Generate deterministic BLAKE2b checksums for every Parquet file under data/."""
import json
import hashlib
from pathlib import Path
from typing import Dict

def parquet_checksum(path: Path) -> str:
    """Return hex BLAKE2b of a Parquet file."""
    h = hashlib.blake2b()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def checksum_stage(stage_dir: Path) -> Dict[str, str]:
    """Checksum all .parquet files in a stage directory."""
    checksums: Dict[str, str] = {}
    for p in sorted(stage_dir.rglob("*.parquet")):
        rel = str(p.relative_to(stage_dir))
        checksums[rel] = parquet_checksum(p)
    return checksums

def main():
    root = Path("data")
    stages = ["raw", "normalized", "aligned", "features", "labels", "splits"]
    manifest: Dict[str, Dict[str, str]] = {}
    for stage in stages:
        stage_dir = root / stage
        if not stage_dir.exists():
            continue
        manifest[stage] = checksum_stage(stage_dir)
    out_path = root / "checksums.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Checksums written to {out_path}")

if __name__ == "__main__":
    main()
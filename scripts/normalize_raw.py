#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.normalize import normalize_symbol, build_normalized_manifest

def main():
    raw_root = Path("data/raw")
    out_root = Path("data/normalized")
    if not raw_root.exists():
        print("No data/raw folder found; skipping normalization.")
        return
    for symbol_dir in sorted(raw_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        print(f"Normalizing {symbol_dir.name}...")
        normalize_symbol(symbol_dir, out_root)
    build_normalized_manifest(out_root)
    print("Normalization complete. Manifest written to data/normalized/manifest.json")

if __name__ == "__main__":
    main()
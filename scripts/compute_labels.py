#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.labels import add_labels, write_labels, build_labels_manifest

def main():
    features_path = Path("data/features/features.parquet")
    if not features_path.exists():
        print("No features found; skipping labels.")
        return
    import pandas as pd
    df = pd.read_parquet(features_path)
    labeled = add_labels(df, horizon=1, threshold=0.0, neutral_band=0.0)
    write_labels(labeled, Path("data/labels"))
    build_labels_manifest(labeled, Path("data/labels"))
    print("Labels added and written to data/labels/labeled.parquet")

if __name__ == "__main__":
    main()
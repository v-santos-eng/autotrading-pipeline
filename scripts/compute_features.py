#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.features import build_features, write_features, build_features_manifest

def main():
    aligned_path = Path("data/aligned/aligned.parquet")
    if not aligned_path.exists():
        print("No aligned data found; skipping features.")
        return
    import pandas as pd
    df = pd.read_parquet(aligned_path)
    feats = build_features(df)
    write_features(feats, Path("data/features"))
    build_features_manifest(feats, Path("data/features"))
    print("Features computed and written to data/features/features.parquet")

if __name__ == "__main__":
    main()
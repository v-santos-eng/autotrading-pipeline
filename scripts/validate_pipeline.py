#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.validation import validate_stage

def main():
    stages = ["normalized", "aligned", "features", "labels"]
    for stage in stages:
        parquet_path = Path(f"data/{stage}/{stage}.parquet")
        if parquet_path.exists():
            validate_stage(parquet_path, stage)
        else:
            print(f"No parquet found for {stage}, skipping validation.")

if __name__ == "__main__":
    main()
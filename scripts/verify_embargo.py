#!/usr/bin/env python3
"""Verify embargo implementation in walk-forward splits."""
import pandas as pd
import json
from pathlib import Path

def main():
    # Load manifest
    manifest_path = Path("data/splits/manifest.json")
    with manifest_path.open() as f:
        manifest = json.load(f)
    
    print("Embargo Verification Report")
    print("=" * 40)
    
    for fold in manifest:
        fold_id = fold["fold"]
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        
        # Calculate embargo gap in minutes
        embargo_gap = (test_start - train_end).total_seconds() / 60
        
        print(f"Fold {fold_id}:")
        print(f"  Train end:   {train_end}")
        print(f"  Test start:  {test_start}")
        print(f"  Embargo gap: {embargo_gap:.1f} minutes")
        print(f"  Train rows:  {fold['train_rows']}")
        print(f"  Test rows:   {fold['test_rows']}")
        print()

if __name__ == "__main__":
    main()
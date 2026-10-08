#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=24, freq="H"),
        "symbol": "AUDUSD",
        "open": np.linspace(0.6700, 0.6730, 24),
        "high": np.linspace(0.6705, 0.6735, 24),
        "low": np.linspace(0.6695, 0.6725, 24),
        "close": np.linspace(0.6702, 0.6729, 24),
        "volume": np.arange(1000, 3400, 100),
        "bid": np.linspace(0.6701, 0.6728, 24),
        "ask": np.linspace(0.6703, 0.6730, 24),
    })
    out_path = Path("data/aligned/aligned.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Sample aligned data written to {out_path}")

if __name__ == "__main__":
    main()
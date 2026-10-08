#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import pytz

def generate_ohlcv(symbol: str, days: int = 5, freq: str = "1min") -> pd.DataFrame:
    """Generate synthetic 1-min FX OHLCV with bid/ask for microstructure features."""
    # Use symbol-specific seed for deterministic but varied data per symbol
    rng = np.random.default_rng(seed=hash(symbol) % 2**31)
    # Use a fixed start time for deterministic results
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = start + pd.Timedelta(days=days)
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    # base price around 1.0 with small random walk (deterministic)
    base = 1.0 + 0.02  # Fixed base price for reproducibility
    returns = rng.standard_normal(len(idx)) * 0.0005
    close = pd.Series(base * np.exp(np.cumsum(returns)), index=idx)
    # derive OHLC
    high = close * (1 + np.abs(rng.standard_normal(len(idx)) * 0.0003))
    low = close * (1 - np.abs(rng.standard_normal(len(idx)) * 0.0003))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = rng.poisson(lam=1000, size=len(idx))
    # microstructure: bid/ask around close
    half_spread = np.abs(rng.standard_normal(len(idx)) * 0.0001)
    bid = close - half_spread
    ask = close + half_spread
    df = pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "bid": bid,
        "ask": ask,
        "volume": volume
    })
    # basic price logic enforcement
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    return df

def main():
    out_root = Path("data/raw")
    symbols = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    for sym in symbols:
        out_dir = out_root / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        df = generate_ohlcv(sym, days=5, freq="1min")
        out_path = out_dir / f"{sym}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
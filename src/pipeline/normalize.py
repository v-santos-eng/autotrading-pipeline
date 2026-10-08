from pathlib import Path
import json
from typing import Dict, List, Optional

import pandas as pd
import pytz
from pandas_market_calendars import get_calendar

CANONICAL_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
CALENDAR_NAME = "24/5"  # 24h Mon-Fri, no holidays; use "NYSE" or "CME" for equities/futures

def read_raw(symbol_dir: Path) -> pd.DataFrame:
    """Read all CSV/Parquet files under symbol_dir and concatenate."""
    dfs = []
    for path in sorted(symbol_dir.rglob("*.csv")) + sorted(symbol_dir.rglob("*.parquet")):
        if path.suffix == ".csv":
            df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            df = pd.read_parquet(path)
        df["symbol"] = symbol_dir.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=CANONICAL_COLS)
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"])
    return df

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical columns and dtypes."""
    df = df.copy()
    for col in CANONICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col}")
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # basic price logic check
    mask = (df["low"] <= df["open"]) & (df["low"] <= df["close"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"])
    if not mask.all():
        raise ValueError("Price logic violation: low <= open/close <= high")
    return df[CANONICAL_COLS].reset_index(drop=True)

def apply_calendar_filter(df: pd.DataFrame, calendar_name: str = CALENDAR_NAME) -> pd.DataFrame:
    """Drop rows outside market calendar sessions."""
    cal = get_calendar(calendar_name)
    schedule = cal.schedule(start_date=df["timestamp"].min().date(), end_date=df["timestamp"].max().date())
    df = df.set_index("timestamp").sort_index()
    # FX calendar is 24h Mon-Fri; filter weekends only for demo
    mask = df.index.dayofweek < 5
    df = df[mask].reset_index()
    return df

def write_normalized(df: pd.DataFrame, output_dir: Path) -> None:
    """Write normalized data as Parquet partitioned by symbol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = df["symbol"].iloc[0]
    out_path = output_dir / f"{symbol}.parquet"
    df.to_parquet(out_path, index=False)

def normalize_symbol(symbol_dir: Path, output_dir: Path) -> None:
    """Full normalization for one symbol."""
    df = read_raw(symbol_dir)
    df = normalize_schema(df)
    df = apply_calendar_filter(df)
    write_normalized(df, output_dir)

def build_normalized_manifest(output_dir: Path) -> None:
    """Write a manifest with row counts and date ranges."""
    rows = []
    for path in sorted(output_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["timestamp"])
        rows.append({
            "symbol": path.stem,
            "rows": len(df),
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
        })
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"normalized": rows}, f)
from pathlib import Path
import json
from typing import Dict, List, Optional

import pandas as pd
import pytz


def resample_bars(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Deterministic resampling to bars with explicit bar-close timestamps.
    freq: pandas offset string, e.g. '1min', '5min', '1h', '1d'
    """
    if df.empty:
        return df
    df = df.set_index("timestamp").sort_index()
    resampled = (
        df.groupby("symbol")
        .resample(freq, closed="right", label="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .reset_index()
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def align_multi_asset(
    dfs: Dict[str, pd.DataFrame], strategy: str = "inner"
) -> pd.DataFrame:
    """
    Align multiple symbols.
    strategy: 'inner' (strict synchronous), 'left' (anchor first symbol), 'outer' (with masks)
    """
    if not dfs:
        return pd.DataFrame()
    symbols = list(dfs.keys())
    
    # Find common timestamps across all symbols
    common_timestamps = dfs[symbols[0]][["timestamp"]].copy()
    for sym in symbols[1:]:
        sym_timestamps = dfs[sym][["timestamp"]].copy()
        common_timestamps = common_timestamps.merge(sym_timestamps, on="timestamp", how="inner")
    
    # Filter each symbol's data to common timestamps and concatenate
    aligned_data = []
    for symbol in symbols:
        symbol_data = dfs[symbol]
        filtered = symbol_data.merge(common_timestamps, on="timestamp", how="inner")
        aligned_data.append(filtered)
    
    aligned = pd.concat(aligned_data, ignore_index=True)
    aligned["is_missing"] = aligned["close"].isna()
    return aligned.reset_index(drop=True)


def write_aligned(df: pd.DataFrame, output_dir: Path) -> None:
    """Write aligned data as single Parquet with symbol partition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aligned.parquet"
    df.to_parquet(out_path, index=False)


def build_aligned_manifest(df: pd.DataFrame, output_dir: Path) -> None:
    """Write manifest with alignment stats."""
    stats = (
        df.groupby("symbol")
        .agg(
            rows=("timestamp", "count"),
            missing=("is_missing", "sum"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        )
        .reset_index()
    )
    # convert timestamps to ISO strings
    stats["start"] = stats["start"].apply(lambda x: x.isoformat())
    stats["end"] = stats["end"].apply(lambda x: x.isoformat())
    records = stats.to_dict(orient="records")
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"aligned": records}, f)


def align_pipeline(
    normalized_dir: Path,
    freq: str = "1min",
    strategy: str = "inner",
    output_dir: Path = Path("data/aligned"),
) -> None:
    """
    Full alignment pipeline: load normalized, resample, align, write.
    """
    dfs = {}
    for path in sorted(normalized_dir.glob("*.parquet")):
        if path.name == "manifest.json":
            continue
        df = pd.read_parquet(path)
        symbol = path.stem
        df = resample_bars(df, freq)
        dfs[symbol] = df
    aligned = align_multi_asset(dfs, strategy)
    write_aligned(aligned, output_dir)
    build_aligned_manifest(aligned, output_dir)
    print(f"Aligned {len(dfs)} symbols at {freq} using {strategy} strategy.")
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    align_pipeline(Path("data/normalized"))
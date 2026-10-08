from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, periods: int = 1, log: bool = True) -> pd.Series:
    """Compute causal returns (log or arithmetic) with lag."""
    close = df["close"]
    if log:
        return np.log(close / close.shift(periods))
    else:
        return close.pct_change(periods)


def rolling_volatility(
    df: pd.DataFrame, window: int = 20, annualize: bool = True, trading_hours: int = 24
) -> pd.Series:
    """Rolling realized volatility using log returns."""
    ret = compute_returns(df, periods=1, log=True)
    vol = ret.rolling(window).std()
    if annualize:
        vol *= np.sqrt(trading_hours * 365)
    return vol


def ewma_volatility(df: pd.DataFrame, span: int = 20, annualize: bool = True) -> pd.Series:
    """Exponentially weighted moving average volatility."""
    ret = compute_returns(df, periods=1, log=True)
    return ret.ewm(span=span).std() * (np.sqrt(365 * 24) if annualize else 1)


def moving_average(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average of close."""
    return df["close"].rolling(window).mean()


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def spread(df: pd.DataFrame) -> pd.Series:
    """Midprice and relative spread if bid/ask present."""
    if {"bid", "ask"}.issubset(df.columns):
        mid = (df["bid"] + df["ask"]) / 2
        spr = (df["ask"] - df["bid"]) / mid
        return spr
    else:
        # fallback: use high/low as proxy
        mid = (df["high"] + df["low"]) / 2
        spr = (df["high"] - df["low"]) / mid
        return spr


def volume_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling z-score of volume relative to its recent mean."""
    vol = df["volume"]
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std()
    return (vol - mean) / std


def cross_sectional_rank(df: pd.DataFrame, feature_col: str) -> pd.Series:
    """Rank within universe at each timestamp (no leakage)."""
    return df.groupby("timestamp")[feature_col].rank(pct=True)


def build_features(
    aligned_df: pd.DataFrame,
    horizon: int = 1,
    vol_window: int = 20,
    rsi_window: int = 14,
    ma_window: int = 20,
) -> pd.DataFrame:
    """
    Main feature builder.
    horizon: forward periods for target (used for lag alignment, not peeking)
    """
    features = aligned_df.copy()
    # returns
    features["ret"] = compute_returns(features, periods=1, log=True)
    features["ret_5"] = compute_returns(features, periods=5, log=True)
    # volatility
    features["vol"] = rolling_volatility(features, window=vol_window)
    features["vol_ewma"] = ewma_volatility(features, span=vol_window)
    # momentum
    features["ma"] = moving_average(features, window=ma_window)
    features["rsi"] = rsi(features, window=rsi_window)
    # microstructure
    features["spread"] = spread(features)
    # volume
    features["vol_imb"] = volume_imbalance(features, window=vol_window)
    # cross-sectional ranks (no leakage)
    for col in ["ret", "vol", "rsi"]:
        features[f"{col}_rank"] = cross_sectional_rank(features, col)
    # drop rows with insufficient history
    min_rows = max(horizon, vol_window, rsi_window, ma_window) + 1
    features = features.dropna(subset=["ret", "vol", "ma", "rsi"])
    return features.reset_index(drop=True)


def write_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Write features as Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features.parquet"
    df.to_parquet(out_path, index=False)


def build_features_manifest(df: pd.DataFrame, output_dir: Path) -> None:
    """Write manifest with feature columns and NaN counts."""
    cols = [c for c in df.columns if c not in {"timestamp", "symbol", "is_missing"}]
    stats = {col: int(df[col].isna().sum()) for col in cols}
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"features": stats}, f)


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    aligned_path = Path("data/aligned/aligned.parquet")
    if not aligned_path.exists():
        print("No aligned data found; skipping features.")
        sys.exit(0)
    df = pd.read_parquet(aligned_path)
    feats = build_features(df)
    write_features(feats, Path("data/features"))
    build_features_manifest(feats, Path("data/features"))
    print("Features computed and written to data/features/features.parquet")
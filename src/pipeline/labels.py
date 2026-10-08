from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int = 1, log: bool = True) -> pd.Series:
    """Compute forward return at horizon periods."""
    close = df["close"]
    future = close.shift(-horizon)
    if log:
        return np.log(future / close)
    else:
        return (future - close) / close


def classify_direction(
    ret: pd.Series, threshold: float = 0.0, neutral_band: float = 0.0
) -> pd.Series:
    """
    Classify returns into {-1, 0, 1}.
    threshold: minimum move to be considered up/down
    neutral_band: symmetric band around zero for neutral class
    """
    y = pd.Series(0, index=ret.index)
    y.loc[ret > threshold + neutral_band] = 1
    y.loc[ret < -threshold - neutral_band] = -1
    return y


def add_labels(
    features_df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0, neutral_band: float = 0.0
) -> pd.DataFrame:
    """Add forward return and direction label columns."""
    df = features_df.copy()
    df["fwd_ret"] = forward_return(df, horizon=horizon, log=True)
    df["label"] = classify_direction(df["fwd_ret"], threshold=threshold, neutral_band=neutral_band)
    return df


def embargo_mask(
    df: pd.DataFrame, embargo: int = 5
) -> pd.Series:
    """
    Create embargo mask: True for rows that should be excluded from training due to overlap.
    Embargo is number of periods after each test fold edge.
    """
    # simple implementation: mark next `embargo` rows after each timestamp as embargoed
    # in practice you'd integrate with your CV splitter
    mask = pd.Series(False, index=df.index)
    # placeholder: no embargo applied here; integrate with your splitter
    return mask


def split_walk_forward(
    df: pd.DataFrame,
    train_size: int = 1000,
    test_size: int = 200,
    step_size: int = 100,
    embargo: int = 5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward splits.
    Returns list of (train_df, test_df) pairs.
    """
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > len(df):
            break
        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        splits.append((train_df, test_df))
        start += step_size
    return splits


def write_labels(df: pd.DataFrame, output_dir: Path) -> None:
    """Write labeled dataset as Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "labeled.parquet"
    df.to_parquet(out_path, index=False)


def build_labels_manifest(df: pd.DataFrame, output_dir: Path) -> None:
    """Write manifest with label distribution."""
    counts = df["label"].value_counts().to_dict()
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"labels": counts}, f)


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    features_path = Path("data/features/features.parquet")
    if not features_path.exists():
        print("No features found; skipping labels.")
        sys.exit(0)
    import pandas as pd
    df = pd.read_parquet(features_path)
    labeled = add_labels(df, horizon=1, threshold=0.0, neutral_band=0.0)
    write_labels(labeled, Path("data/labels"))
    build_labels_manifest(labeled, Path("data/labels"))
    print("Labels added and written to data/labels/labeled.parquet")
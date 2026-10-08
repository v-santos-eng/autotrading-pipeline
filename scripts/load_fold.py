#!/usr/bin/env python3
"""Load a single walk-forward fold and split into X/y."""
import pandas as pd
from pathlib import Path
from typing import Tuple

def load_fold(fold: int, splits_dir: Path = Path("data/splits")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for a given fold index."""
    train_path = splits_dir / f"train_fold_{fold}.parquet"
    test_path = splits_dir / f"test_fold_{fold}.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Fold {fold} not found in {splits_dir}")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df

def feature_label_split(df: pd.DataFrame, label_col: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X (features) and y (label)."""
    if label_col not in df.columns:
        # fallback: use last column as label
        label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y

if __name__ == "__main__":
    # quick sanity check
    train_df, test_df = load_fold(0)
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
    X_train, y_train = feature_label_split(train_df)
    print("X_train shape:", X_train.shape, "y_train distribution:")
    print(y_train.value_counts())
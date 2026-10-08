from pathlib import Path
import json
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


def check_schema(df: pd.DataFrame, required: List[str]) -> Dict[str, bool]:
    """Check presence and dtypes of required columns."""
    ok = {}
    for col in required:
        ok[col] = col in df.columns
        if ok[col] and col == "timestamp":
            ok["timestamp_dtype"] = pd.api.types.is_datetime64_any_dtype(df[col])
    return ok


def check_price_logic(df: pd.DataFrame) -> Dict[str, bool]:
    """Check low <= open/close <= high and non-negative volume."""
    ok = {}
    ok["low_leq_open"] = (df["low"] <= df["open"]).all()
    ok["low_leq_close"] = (df["low"] <= df["close"]).all()
    ok["high_geq_open"] = (df["high"] >= df["open"]).all()
    ok["high_geq_close"] = (df["high"] >= df["close"]).all()
    ok["volume_nonneg"] = (df["volume"] >= 0).all()
    return ok


def check_time_monotonic(df: pd.DataFrame) -> Dict[str, bool]:
    """Check timestamp is monotonic per symbol."""
    ok = {}
    for sym, g in df.groupby("symbol"):
        ok[f"monotonic_{sym}"] = g["timestamp"].is_monotonic_increasing
    return ok


def check_missingness(df: pd.DataFrame, max_missing: float = 0.05) -> Dict[str, bool]:
    """Check missingness rates per column."""
    ok = {}
    for col in df.columns:
        rate = df[col].isna().mean()
        ok[f"missing_{col}"] = rate <= max_missing
    return ok


def check_alignment(df: pd.DataFrame) -> Dict[str, bool]:
    """Check alignment masks are present and consistent."""
    ok = {}
    if "is_missing" in df.columns:
        ok["missing_mask"] = df["is_missing"].dtype == bool
        ok["any_missing"] = df["is_missing"].any()
    return ok


def check_features(df: pd.DataFrame, required: List[str]) -> Dict[str, bool]:
    """Check feature columns exist and no NaNs after warm-up."""
    ok = {}
    for col in required:
        ok[f"feature_{col}"] = col in df.columns
        if ok[f"feature_{col}"]:
            ok[f"nan_{col}"] = df[col].isna().sum() == 0
    return ok


def check_labels(df: pd.DataFrame) -> Dict[str, bool]:
    """Check label distribution and horizon shift."""
    ok = {}
    if "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        ok["label_counts"] = len(counts) >= 2
        ok["label_balanced"] = min(counts.values()) / max(counts.values()) >= 0.1
    if "fwd_ret" in df.columns:
        ok["fwd_ret_finite"] = np.isfinite(df["fwd_ret"]).all()
    return ok


def run_validation(
    df: pd.DataFrame,
    stage: str,
    required_cols: Optional[List[str]] = None,
    max_missing: float = 0.05,
) -> Dict[str, Dict[str, bool]]:
    """Run all validation checks for a given stage."""
    checks = {}
    checks["schema"] = check_schema(df, required_cols or ["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    checks["price_logic"] = check_price_logic(df)
    checks["time_monotonic"] = check_time_monotonic(df)
    checks["missingness"] = check_missingness(df, max_missing)
    checks["alignment"] = check_alignment(df)
    if stage in {"features", "labels"}:
        checks["features"] = check_features(df, ["ret", "vol", "ma", "rsi"])
    if stage == "labels":
        checks["labels"] = check_labels(df)
    return checks


def write_validation_report(checks: Dict[str, Dict[str, bool]], output_path: Path) -> None:
    """Write validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # convert numpy bools to native Python bools
    serializable = {}
    for group, items in checks.items():
        serializable[group] = {k: bool(v) for k, v in items.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def validate_stage(
    parquet_path: Path,
    stage: str,
    required_cols: Optional[List[str]] = None,
    max_missing: float = 0.05,
) -> None:
    """Load parquet and validate."""
    df = pd.read_parquet(parquet_path)
    checks = run_validation(df, stage, required_cols, max_missing)
    report_path = parquet_path.parent / f"validation_{stage}.json"
    write_validation_report(checks, report_path)
    print(f"Validation report written to {report_path}")
    # print summary
    for group, items in checks.items():
        failed = [k for k, v in items.items() if not v]
        if failed:
            print(f"{group}: FAIL {failed}")
        else:
            print(f"{group}: OK")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pathlib import Path

    stages = ["normalized", "aligned", "features", "labels"]
    for stage in stages:
        parquet_path = Path(f"data/{stage}/{stage}.parquet")
        if parquet_path.exists():
            validate_stage(parquet_path, stage)
        else:
            print(f"No parquet found for {stage}, skipping validation.")
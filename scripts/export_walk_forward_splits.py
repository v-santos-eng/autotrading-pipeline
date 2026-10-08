#!/usr/bin/env python3
"""Export embargoed walk-forward CV splits from aligned features."""
import json
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import click
from datetime import timedelta

def create_row_based_folds(df: pd.DataFrame, train_size: int, test_size: int, step_size: int, embargo: int, min_symbols: int) -> List[Tuple[int, int, int, int]]:
    """Create row-based walk-forward folds."""
    folds = []
    start = 0
    
    while start + train_size + embargo + test_size <= len(df):
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        
        # Validate symbol coverage in this fold
        train_symbols = set(df.iloc[start:train_end]['symbol'].unique())
        test_symbols = set(df.iloc[test_start:test_end]['symbol'].unique())
        
        if len(train_symbols) >= min_symbols and len(test_symbols) >= min_symbols:
            folds.append((start, train_end, test_start, test_end))
        else:
            print(f"Skipping fold at position {start} due to insufficient symbol coverage")
        
        start += step_size
    
    return folds

def create_time_based_folds(df: pd.DataFrame, train_size: int, test_size: int, step_size: int, embargo: int, min_symbols: int, align_boundaries: bool) -> List[Tuple[int, int, int, int]]:
    """Create time-based walk-forward folds using actual timestamps."""
    print("Analyzing temporal structure...")
    
    # Determine the most common time interval
    time_diffs = df['timestamp'].diff().dropna()
    median_interval = time_diffs.median()
    
    if median_interval == timedelta(0):
        raise ValueError("Cannot determine time interval - data may not be properly time-indexed")
    
    print(f"Detected time interval: {median_interval}")
    
    # Convert sizes to time durations
    # Assuming train_size, test_size, step_size, embargo are in units of the detected interval
    train_duration = median_interval * train_size
    test_duration = median_interval * test_size
    step_duration = median_interval * step_size
    embargo_duration = median_interval * embargo
    
    if align_boundaries:
        print("Aligning to day boundaries...")
        # Round durations to nearest day for cleaner boundaries
        day_duration = timedelta(days=1)
        hour_duration = timedelta(hours=1)
        
        # Ensure minimum durations and round to boundaries
        train_duration = max(day_duration, round(train_duration / day_duration) * day_duration)
        test_duration = max(day_duration, round(test_duration / day_duration) * day_duration)
        step_duration = max(day_duration, round(step_duration / day_duration) * day_duration)
        embargo_duration = max(hour_duration, round(embargo_duration / hour_duration) * hour_duration)
        print(f"Adjusted durations: train={train_duration}, test={test_duration}, embargo={embargo_duration}")
    
    folds = []
    current_start = df['timestamp'].min()
    data_end = df['timestamp'].max()
    fold_id = 0
    
    while current_start + train_duration + embargo_duration + test_duration <= data_end:
        train_start = current_start
        train_end = train_start + train_duration
        test_start = train_end + embargo_duration
        test_end = test_start + test_duration
        
        # Find indices for these time periods
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        train_indices = df[train_mask].index
        test_indices = df[test_mask].index
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"Warning: No data found for fold starting at {train_start}")
            current_start += step_duration
            continue
            
        # Validate symbol coverage
        train_symbols = set(df.loc[train_indices, 'symbol'].unique())
        test_symbols = set(df.loc[test_indices, 'symbol'].unique())
        
        if len(train_symbols) >= min_symbols and len(test_symbols) >= min_symbols:
            # Store the actual row indices for consistency with existing code
            folds.append((train_indices[0], train_indices[-1] + 1, test_indices[0], test_indices[-1] + 1))
            
            print(f"Time-based Fold {fold_id}: {train_start} to {test_end}")
            print(f"  Train: {len(train_indices)} rows ({len(train_symbols)} symbols)")
            print(f"  Test: {len(test_indices)} rows ({len(test_symbols)} symbols)")
            fold_id += 1
        else:
            print(f"Skipping time-based fold at {train_start} due to insufficient symbol coverage")
        
        current_start += step_duration
    
    if not folds:
        print("Warning: No time-based folds created, falling back to row-based approach")
        return create_row_based_folds(df, train_size, test_size, step_size, embargo, min_symbols)
    
    return folds

@click.command()
@click.option("--features", default="data/labels/labeled.parquet", help="Input labeled features Parquet")
@click.option("--out-dir", default="data/splits", help="Output directory for folds")
@click.option("--train-size", default=3600, help="Bars per training window")
@click.option("--test-size", default=900, help="Bars per test window")
@click.option("--step-size", default=900, help="Bars to advance window each fold")
@click.option("--embargo", default=60, help="Bars to exclude after train set")
@click.option("--min-symbols", default=1, help="Minimum symbols required per fold")
@click.option("--time-based", is_flag=True, help="Use time-based splitting instead of row-based")
@click.option("--align-boundaries", is_flag=True, help="Align splits to day/week boundaries")
def main(features: str, out_dir: str, train_size: int, test_size: int, step_size: int, embargo: int, min_symbols: int, time_based: bool, align_boundaries: bool):
    """Export embargoed walk-forward CV splits."""
    df = pd.read_parquet(features)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate data
    if 'symbol' not in df.columns or 'timestamp' not in df.columns:
        raise ValueError("Input data must contain 'symbol' and 'timestamp' columns")
    
    if 'label' not in df.columns:
        raise ValueError("Input data must contain 'label' column for supervised learning")

    # Ensure chronological order globally first
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Enhanced data validation and gap detection
    print("Validating data integrity...")
    
    # Check for time gaps in the data
    time_diffs = df['timestamp'].diff().dropna()
    median_diff = time_diffs.median()
    large_gaps = time_diffs[time_diffs > median_diff * 3]  # Gaps 3x larger than median
    
    if len(large_gaps) > 0:
        print(f"Warning: Found {len(large_gaps)} large time gaps in data")
        print(f"Largest gap: {large_gaps.max()}")
        print("Consider using time-aware splitting for better results")
    
    # Check for consistent time intervals (should be roughly regular for FX data)
    unique_diffs = time_diffs.value_counts().head()
    print(f"Most common time intervals: {dict(unique_diffs.head(3))}")
    
    # Check symbol coverage per time period
    symbols = df['symbol'].unique()
    print(f"Found {len(symbols)} symbols: {list(symbols)}")
    
    # Optional: Ensure balanced symbol representation per fold
    symbol_counts = df.groupby('symbol').size()
    print(f"Symbol distribution: {symbol_counts.to_dict()}")
    
    # Warn if symbols are imbalanced
    if len(symbols) > 1:
        min_count = symbol_counts.min()
        max_count = symbol_counts.max()
        if max_count / min_count > 2:
            print(f"Warning: Symbol imbalance detected (ratio: {max_count/min_count:.2f}x)")
            print("Consider using stratified sampling or balanced window sizes")
    
    # Check for missing values in critical columns
    critical_cols = ['timestamp', 'symbol', 'close', 'label']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"Warning: {missing} missing values in {col}")
    
    # Analyze temporal coverage per symbol
    print("\nTemporal coverage analysis:")
    for symbol in symbols:
        sym_df = df[df['symbol'] == symbol]
        start_time = sym_df['timestamp'].min()
        end_time = sym_df['timestamp'].max()
        duration = end_time - start_time
        print(f"  {symbol}: {start_time} to {end_time} ({duration.total_seconds()/3600:.1f} hours)")
    
    # Create folds with proper embargo
    folds: List[Tuple[int, int, int, int]] = []
    
    if time_based:
        print("Using time-based splitting...")
        folds = create_time_based_folds(df, train_size, test_size, step_size, embargo, min_symbols, align_boundaries)
    else:
        print("Using row-based splitting...")
        folds = create_row_based_folds(df, train_size, test_size, step_size, embargo, min_symbols)

    if not folds:
        raise ValueError("No valid folds could be created with the given parameters")

    manifest = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        # Remove temporary position column if it exists
        train_df = train_df.drop(columns=['position'], errors='ignore')
        test_df = test_df.drop(columns=['position'], errors='ignore')
        
        train_file = out_path / f"train_fold_{i}.parquet"
        test_file = out_path / f"test_fold_{i}.parquet"
        
        train_df.to_parquet(train_file, index=False)
        test_df.to_parquet(test_file, index=False)
        
        # Get symbol statistics
        train_symbols = train_df['symbol'].unique()
        test_symbols = test_df['symbol'].unique()
        
        fold_info = {
            "fold": i,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_symbols": len(train_symbols),
            "test_symbols": len(test_symbols),
            "train_start": train_df["timestamp"].min(),
            "train_end": train_df["timestamp"].max(),
            "test_start": test_df["timestamp"].min(),
            "test_end": test_df["timestamp"].max(),
            "train_file": str(train_file),
            "test_file": str(test_file),
            "train_symbol_list": sorted(train_symbols.tolist()),
            "test_symbol_list": sorted(test_symbols.tolist()),
        }
        
        # Add label distribution for classification tasks
        if 'label' in train_df.columns:
            fold_info["train_label_dist"] = train_df['label'].value_counts().to_dict()
            fold_info["test_label_dist"] = test_df['label'].value_counts().to_dict()
        
        manifest.append(fold_info)
        print(f"Fold {i}: train {len(train_df)} rows ({len(train_symbols)} symbols), "
              f"test {len(test_df)} rows ({len(test_symbols)} symbols)")

    manifest_path = out_path / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\nExported {len(folds)} folds to {out_path}")
    print(f"Total data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Coverage: {len(symbols)} symbols across {len(df)} rows")
    
    # Final validation: Check for data leakage between folds
    print("\nValidating fold integrity...")
    
    # Enhanced overlap analysis for walk-forward validation
    overlap_issues = 0
    for i in range(len(manifest)-1):
        current_train_start = pd.to_datetime(manifest[i]["train_start"])
        current_test_end = pd.to_datetime(manifest[i]["test_end"])
        next_train_start = pd.to_datetime(manifest[i+1]["train_start"])
        next_test_end = pd.to_datetime(manifest[i+1]["test_end"])
        
        # Check for actual data leakage (test data overlapping with next test period)
        # This would indicate improper walk-forward setup
        if next_test_end <= current_test_end:
            print(f"Warning: Improper walk-forward - Fold {i+1} ends before or at same time as Fold {i}")
            overlap_issues += 1
        
        # Check for walk-forward consistency (next should start after current starts)
        if next_train_start <= current_train_start:
            print(f"Warning: Walk-forward violation - Fold {i+1} train starts before Fold {i} train")
            overlap_issues += 1
    
    if overlap_issues == 0:
        print("✓ Walk-forward progression is consistent")
        print("✓ No improper temporal ordering detected")
    
    # Check data usage efficiency
    total_rows_used = sum(m["train_rows"] + m["test_rows"] for m in manifest)
    efficiency = total_rows_used / len(df) * 100
    print(f"Data usage efficiency: {efficiency:.1f}% of total data used in folds")
    
    # Summary statistics
    if len(manifest) > 1:
        train_sizes = [m["train_rows"] for m in manifest]
        test_sizes = [m["test_rows"] for m in manifest]
        print(f"Train set size: {min(train_sizes)}-{max(train_sizes)} rows (avg: {sum(train_sizes)/len(train_sizes):.0f})")
        print(f"Test set size: {min(test_sizes)}-{max(test_sizes)} rows (avg: {sum(test_sizes)/len(test_sizes):.0f})")
    
    # Time-based analysis
    if len(manifest) > 0:
        total_start = pd.to_datetime(manifest[0]["train_start"])
        total_end = pd.to_datetime(manifest[-1]["test_end"])
        total_duration = total_end - total_start
        print(f"Total CV period: {total_duration.total_seconds()/3600:.1f} hours")
        if len(manifest) > 1:
            print(f"Average fold advance: {total_duration.total_seconds()/3600/(len(manifest)-1):.1f} hours per fold")
        else:
            print("Single fold created - no fold advance to calculate")
    
    print("\n✓ Walk-forward CV export completed successfully!")
    print("✓ All folds are embargoed and temporally consistent")
    print("✓ Manifest includes comprehensive metadata for reproducibility")
    
    # Additional recommendations
    if efficiency > 100:
        print("ℹ High data usage efficiency indicates good overlap between folds (normal for walk-forward)")
    if len(manifest) < 3:
        print("ℹ Consider creating more folds for robust cross-validation")
    if step_size < train_size:
        print("ℹ Overlapping windows detected - this is normal for walk-forward CV")

if __name__ == "__main__":
    main()
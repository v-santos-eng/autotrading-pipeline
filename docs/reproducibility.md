# Deterministic Reproducibility & Walk-Forward CV Export

## Checksums

Generate and verify deterministic checksums for all Parquet artifacts:

```bash
python scripts/checksums.py
```

This writes `checksums.json` under each stage directory and verifies integrity on re-run.

## Deterministic Data Generation

The sample generator now uses a fixed seed (42) to ensure identical raw data across runs. Change the seed in `scripts/generate_sample_raw.py` if you need a different sample.

## Walk-Forward CV Splits

Export embargoed, non-overlapping folds:

```bash
python scripts/export_walk_forward_splits.py
```

This creates `data/splits/train_fold_{n}.parquet` and `test_fold_{n}.parquet` with configurable:
- `train_size`: number of bars per training window
- `test_size`: number of bars per test window
- `step_size`: bars to advance the window each fold
- `embargo`: bars to exclude after each training set to prevent leakage

## Loading a Fold

```python
from scripts.load_fold import load_fold, feature_label_split

train_df, test_df = load_fold(0)
X_train, y_train = feature_label_split(train_df)
X_test, y_test = feature_label_split(test_df)
```

## Reproducibility Checklist

- [x] Fixed random seeds in data generation (symbol-specific seeds for varied but deterministic data)
- [x] Deterministic resampling and alignment rules
- [x] Immutable Parquet outputs with BLAKE2b checksums
- [x] Walk-forward splits with embargo (60-minute embargo period implemented)
- [x] Versioned manifests capturing parameters and hashes
- [x] Load_fold utility for easy fold loading
- [x] Deterministic checksums for all data stages (raw, normalized, aligned, features, labels, splits)

All artifacts under `data/` are now deterministic and verifiable.

## Quick Start

1. Generate synthetic data: `python scripts/generate_sample_raw.py`
2. Run full pipeline: `python scripts/run_pipeline.py`
3. Compute checksums: `python scripts/checksums.py`
4. Export CV splits: `python scripts/export_walk_forward_splits.py`
5. Load a fold: `python scripts/load_fold.py`

All scripts are idempotent and will overwrite previous outputs while maintaining deterministic results.
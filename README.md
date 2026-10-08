# Autotrading Pipeline

High-integrity autotrading dataset pipeline ensuring temporal consistency, statistical robustness, and zero information leakage.

## Features

- **Leakage-safe causal feature engineering** - All features use only past data
- **Multi-asset time-series alignment** - Configurable join strategies (inner/left/outer)
- **Walk-forward cross-validation** - With embargo and purging to prevent data leakage
- **Deterministic reproducibility** - Fixed seeds and checksums for audit trails
- **Comprehensive validation** - Schema, price logic, time consistency, and missingness checks
- **Parquet-based storage** - Immutable artifacts with versioned manifests

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

Generate synthetic FX data and run the full pipeline:

```bash
# Run complete pipeline
autotrading pipeline

# Or run individual steps
autotrading generate      # Generate synthetic 1-min OHLCV data
autotrading normalize     # Normalize and calendar-filter
autotrading align        # Align multi-asset time series
autotrading features     # Compute causal features
autotrading labels       # Add forward returns and classification
autotrading validate     # Validate all pipeline outputs
```

## Pipeline Architecture

```
raw/               →  normalized/      →  aligned/        →  features/       →  labels/
1-min OHLCV         Calendar-filtered    Multi-asset        Causal features    Forward returns
                    Timezone-fixed       aligned            No look-ahead      Classification
```

## Configuration

The pipeline uses sensible defaults but can be customized via:
- Command-line arguments (see `--help` on any script)
- JSON configuration files in `data/configs/`
- Environment variables for paths and parameters

## Validation

Each pipeline stage includes comprehensive validation:
- **Schema validation** - Column types and required fields
- **Price logic checks** - OHLC relationships, no negative prices
- **Time consistency** - Monotonic timestamps, no duplicates
- **Missingness analysis** - Per-column statistics and patterns

## Walk-Forward Cross-Validation

Generate embargoed CV splits for robust backtesting:

```bash
autotrading splits --folds 5 --test-days 30 --embargo-days 5
```

This creates train/test splits with:
- **Embargo periods** - Prevent look-ahead bias
- **Purging** - Remove overlapping samples
- **Deterministic splits** - Reproducible across runs

## Data Format

### Input (Raw)
- 1-minute OHLCV with bid/ask quotes
- UTC timestamps
- Symbol-specific directories

### Output (Features)
- Returns, volatility, RSI, moving averages
- Bid-ask spread and volume imbalance
- Forward return labels with classification
- All features are causal (no future data)

## Development

Run tests:
```bash
pytest tests/
```

Type checking:
```bash
mypy src/
```

Code formatting:
```bash
black scripts/ src/ tests/
```

## Status

- The Python pipeline under `src/pipeline/` and `scripts/` is the working part: it runs end to end on synthetic data and has a pytest suite.
- On the bundled synthetic data the alignment check reports `any_missing`, because the generated symbols do not share every timestamp. That is the validator doing its job, not a crash.
- `src/core`, `src/agents`, `packages/` and `apps/` hold an earlier TypeScript agent-runner scaffold that the pipeline does not depend on.
- No market data or trained models are included; `data/` is generated locally.

## License

MIT License - see LICENSE file for details.
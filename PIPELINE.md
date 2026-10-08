# FX Data Pipeline

## Overview
End-to-end pipeline for FX data processing: raw → normalized → aligned → features → inventory.

## Usage

### Run full pipeline
```bash
python -m agents.cli pipeline
```

### Run individual steps
```bash
# Generate synthetic raw 1-min OHLCV for 7 FX symbols
python scripts/generate_sample_raw.py

# Normalize (calendar filter, resample, clean)
python scripts/normalize_raw.py

# Align symbols to common timestamps
python scripts/align_symbols.py

# Compute features (returns, vol, RSI, MA, spread, volume imbalance, ranks)
python scripts/compute_features.py

# Inventory existing trained models
python scripts/inventory_models.py
```

## Outputs
- `data/raw/`          : synthetic 1-min OHLCV (7 symbols, 5 days, ~7k rows each)
- `data/normalized/`   : calendar-filtered 1-min bars
- `data/aligned/`      : aligned across symbols (inner join)
- `data/features/`     : computed features ( parquet + manifest.json )
- `data/manifests/`    : model inventory (JSON)

## Features computed
- `ret` / `ret_5`      : 1-step and 5-step log returns
- `vol` / `vol_ewma`   : rolling 20-period realized and EWMA volatility (annualized)
- `ma`                 : 20-period simple moving average
- `rsi`                : 14-period Relative Strength Index
- `spread`             : relative bid-ask spread (or high-low proxy)
- `vol_imb`            : volume z-score vs 20-period mean
- `*_rank`             : cross-sectional ranks (timestamp-wise, no leakage)

## CLI quick start
```bash
# List available agents
python -m agents.cli list

# Run echo agent
python -m agents.cli run echo "hello" --params '{}'

# Run full FX pipeline
python -m agents.cli pipeline
```
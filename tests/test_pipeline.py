import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline.normalize import normalize_symbol, build_normalized_manifest
from src.pipeline.align import align_pipeline
from src.pipeline.features import build_features, compute_returns, rolling_volatility, rsi
from src.pipeline.labels import add_labels, forward_return
from src.pipeline.validation import validate_stage, run_validation


class TestNormalize:
    """Test data normalization pipeline."""
    
    def test_normalize_symbol(self):
        """Test basic symbol normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            raw_dir = tmpdir / "raw" / "TEST"
            raw_dir.mkdir(parents=True)
            
            # Generate synthetic data
            timestamps = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": 1.0 + np.random.randn(100) * 0.001,
                "high": 1.001 + np.random.randn(100) * 0.001,
                "low": 0.999 + np.random.randn(100) * 0.001,
                "close": 1.0 + np.random.randn(100) * 0.001,
                "bid": 1.0 + np.random.randn(100) * 0.0005,
                "ask": 1.0001 + np.random.randn(100) * 0.0005,
                "volume": np.random.poisson(1000, 100)
            })
            
            # Ensure price logic
            df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
            df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
            
            df.to_parquet(raw_dir / "TEST.parquet", index=False)
            
            # Test normalization
            out_root = tmpdir / "normalized"
            normalize_symbol(raw_dir, out_root)
            
            assert (out_root / "TEST.parquet").exists()
            
            # Load and verify
            normalized_df = pd.read_parquet(out_root / "TEST.parquet")
            assert len(normalized_df) == len(df)
            assert "timestamp" in normalized_df.columns
            assert normalized_df["timestamp"].dt.tz is not None  # Should be timezone-aware


class TestFeatures:
    """Test feature engineering pipeline."""
    
    def test_compute_returns(self):
        """Test return calculation."""
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 103],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1min")
        })
        returns = compute_returns(df, periods=1, log=False)
        
        assert len(returns) == len(df)
        assert pd.isna(returns.iloc[0])  # First value should be NaN
        assert abs(returns.iloc[1] - 0.01) < 1e-10  # 1% return (with floating point tolerance)
        assert abs(returns.iloc[2] - (102-101)/101) < 1e-10  # ~0.99% (with floating point tolerance)
    
    def test_compute_volatility(self):
        """Test volatility calculation."""
        # Create a DataFrame with close prices
        timestamps = pd.date_range("2025-01-01", periods=100, freq="1min")
        df = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(100) * 0.01),
            "timestamp": timestamps
        })
        volatility = rolling_volatility(df, window=20, annualize=False)
        
        assert len(volatility) == len(df)
        assert pd.isna(volatility.iloc[:20]).all()  # First 20 values should be NaN
        assert volatility.iloc[20:].notna().all()  # Rest should be valid
        assert (volatility.dropna() >= 0).all()  # Volatility should be non-negative (ignoring NaN)
    
    def test_compute_rsi(self):
        """Test RSI calculation."""
        # Create a DataFrame with close prices
        timestamps = pd.date_range("2025-01-01", periods=50, freq="1min")
        df = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(50) * 0.1),
            "timestamp": timestamps
        })
        rsi_values = rsi(df, window=14)
        
        assert len(rsi_values) == len(df)
        # RSI should be valid after the window period (with some buffer for smoothing)
        assert rsi_values.dropna().iloc[14:].between(0, 100).all()  # RSI should be between 0-100


class TestLabels:
    """Test label generation pipeline."""
    
    def test_compute_forward_returns(self):
        """Test forward return calculation."""
        # Create a DataFrame with close prices
        timestamps = pd.date_range("2025-01-01", periods=8, freq="1min")
        df = pd.DataFrame({
            "close": [100, 101, 102, 101, 103, 104, 102, 105],
            "timestamp": timestamps
        })
        forward_returns = forward_return(df, horizon=2, log=False)
        
        assert len(forward_returns) == len(df)
        # Last 2 values should be NaN (no future data)
        assert pd.isna(forward_returns.iloc[-2:]).all()
        # Check specific calculations
        assert forward_returns.iloc[0] == (102-100)/100  # 2-period return from first price
        assert forward_returns.iloc[1] == (101-101)/101  # 2-period return from second price


class TestValidation:
    """Test data validation pipeline."""
    
    def test_validate_stage_valid_data(self):
        """Test validation with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create valid test data
            timestamps = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
            df = pd.DataFrame({
                "timestamp": timestamps,
                "symbol": "TEST",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000
            })
            
            parquet_path = tmpdir / "test.parquet"
            df.to_parquet(parquet_path, index=False)
            
            # Test validation
            checks = run_validation(df, "test")
            
            # Check that all validation groups passed
            all_passed = all(all(checks[group].values()) for group in checks)
            assert all_passed
    
    def test_validate_stage_invalid_prices(self):
        """Test validation with invalid price data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create invalid test data (low > high)
            timestamps = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
            df = pd.DataFrame({
                "timestamp": timestamps,
                "symbol": "TEST",
                "open": 100.0,
                "high": 99.0,  # Invalid: high < low
                "low": 101.0,  # Invalid: low > high
                "close": 100.5,
                "volume": 1000
            })
            
            parquet_path = tmpdir / "test.parquet"
            df.to_parquet(parquet_path, index=False)
            
            # Test validation
            checks = run_validation(df, "test")
            
            # Check that price logic validation failed
            assert "price_logic" in checks
            price_checks = checks["price_logic"]
            assert any(not passed for passed in price_checks.values())


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    def test_pipeline_flow(self):
        """Test basic pipeline flow with synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create raw data directory structure
            raw_dir = tmpdir / "raw" / "TEST"
            raw_dir.mkdir(parents=True)
            
            # Generate synthetic data
            timestamps = pd.date_range("2025-01-01", periods=200, freq="1min", tz="UTC")
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": 1.0 + np.cumsum(np.random.randn(200) * 0.001),
                "high": 1.001 + np.cumsum(np.random.randn(200) * 0.001),
                "low": 0.999 + np.cumsum(np.random.randn(200) * 0.001),
                "close": 1.0 + np.cumsum(np.random.randn(200) * 0.001),
                "bid": 0.9999 + np.cumsum(np.random.randn(200) * 0.0005),
                "ask": 1.0001 + np.cumsum(np.random.randn(200) * 0.0005),
                "volume": np.random.poisson(1000, 200)
            })
            
            # Ensure price logic
            df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
            df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
            
            df.to_parquet(raw_dir / "TEST.parquet", index=False)
            
            # Run normalization
            normalized_dir = tmpdir / "normalized"
            normalize_symbol(raw_dir, normalized_dir)
            
            # Verify normalization worked
            assert (normalized_dir / "TEST.parquet").exists()
            normalized_df = pd.read_parquet(normalized_dir / "TEST.parquet")
            assert len(normalized_df) == len(df)
            
            # Run validation on normalized data
            normalized_df = pd.read_parquet(normalized_dir / "TEST.parquet")
            checks = run_validation(normalized_df, "normalized")
            
            # Check that all validation groups passed
            all_passed = all(all(checks[group].values()) for group in checks)
            assert all_passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
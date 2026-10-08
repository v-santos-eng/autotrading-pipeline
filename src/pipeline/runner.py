from pathlib import Path
from typing import Dict, Optional
import json


class PipelineConfig:
    """Configuration for the autotrading dataset pipeline."""

    def __init__(self, config_path: Optional[Path] = None):
        self.raw_root = Path("data/raw")
        self.normalized_root = Path("data/normalized")
        self.aligned_root = Path("data/aligned")
        self.features_root = Path("data/features")
        self.labels_root = Path("data/labels")
        self.manifests_root = Path("data/manifests")

        # alignment
        self.resample_freq = "1min"
        self.align_strategy = "inner"  # inner, left, outer

        # features
        self.horizon = 1
        self.vol_window = 20
        self.rsi_window = 14
        self.ma_window = 20

        # labels
        self.threshold = 0.0
        self.neutral_band = 0.0

        # validation
        self.max_missing_rate = 0.05

        if config_path and config_path.exists():
            self.load(config_path)

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in dir(self) if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def run_pipeline(config: PipelineConfig) -> None:
    """Run full pipeline with given config."""
    import subprocess
    import sys

    def run_script(name: str) -> None:
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / name
        subprocess.run([sys.executable, str(script)], check=True)

    # normalization
    run_script("normalize_raw.py")
    # alignment
    run_script("align_symbols.py")
    # features
    run_script("compute_features.py")
    # labels
    run_script("compute_labels.py")
    # validation
    run_script("validate_pipeline.py")

    print("Pipeline complete. Artifacts saved under data/")


if __name__ == "__main__":
    cfg_path = Path("pipeline_config.json")
    if not cfg_path.exists():
        cfg = PipelineConfig()
        cfg.save(cfg_path)
        print(f"Created default config at {cfg_path}")
    else:
        cfg = PipelineConfig(cfg_path)
    run_pipeline(cfg)
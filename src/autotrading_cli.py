#!/usr/bin/env python3
"""Console entry point for autotrading pipeline CLI commands."""

import sys
import subprocess
from pathlib import Path

def main():
    """Dispatch to appropriate script based on command name."""
    if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h', 'help']:
        print("AutoTrading Pipeline CLI v0.1.0")
        print("High-integrity FX data pipeline with walk-forward cross-validation")
        print("\nUsage: autotrading <command> [args...]")
        print("\nAvailable commands:")
        print("  generate    - Generate synthetic FX data")
        print("  normalize   - Normalize raw data")
        print("  align       - Align symbols temporally")
        print("  features    - Compute technical features")
        print("  labels      - Generate target labels")
        print("  validate    - Validate pipeline integrity")
        print("  pipeline    - Run complete pipeline")
        print("  splits      - Export walk-forward CV splits")
        print("  checksums   - Compute data integrity checksums")
        print("  load-fold   - Load and validate CV folds")
        print("  train       - Train models on CV splits")
        print("  evaluate    - Evaluate and compare models")
        print("\nUse 'autotrading <command> --help' for command-specific help")
        print("\nExamples:")
        print("  autotrading pipeline                    # Run complete pipeline")
        print("  autotrading splits --train-size 3600    # Create CV splits")
        print("  autotrading train                       # Train models on CV splits")
        print("  autotrading evaluate --comprehensive-report  # Evaluate models")
        print("  autotrading checksums                   # Verify data integrity")
        print("\nFor more information, visit: https://github.com/your-repo/autotrading-pipeline")
        sys.exit(0)
    
    if sys.argv[1] in ['--version', '-v', 'version']:
        print("AutoTrading Pipeline CLI v0.1.0")
        print("High-integrity FX data pipeline with walk-forward cross-validation")
        print("Built with reproducibility and temporal consistency in mind")
        sys.exit(0)
    
    command = sys.argv[1]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    
    # Map commands to scripts
    command_map = {
        "generate": "generate_sample_raw.py",
        "normalize": "normalize_raw.py", 
        "align": "align_symbols.py",
        "features": "compute_features.py",
        "labels": "compute_labels.py",
        "validate": "validate_pipeline.py",
        "pipeline": "run_pipeline.py",
        "splits": "export_walk_forward_splits.py",
        "checksums": "checksums.py",
        "load-fold": "load_fold.py",
        "train": "train_models.py",
        "evaluate": "evaluate_models.py"
    }
    
    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands:", ", ".join(command_map.keys()))
        sys.exit(1)
    
    script_name = command_map[command]
    script_path = scripts_dir / script_name
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(1)
    
    # Run the script with remaining arguments
    cmd = [sys.executable, str(script_path)] + sys.argv[2:]
    result = subprocess.run(cmd, cwd=str(repo_root))
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
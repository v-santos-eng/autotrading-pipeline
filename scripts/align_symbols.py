#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.align import align_pipeline

def main():
    align_pipeline(Path("data/normalized"))

if __name__ == "__main__":
    main()
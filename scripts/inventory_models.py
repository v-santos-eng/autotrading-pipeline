#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline.inventory import collect_models_metadata, write_manifest


def main():
    models_root = Path("data/external/models")
    output_manifest = Path("data/manifests/models_inventory.json")
    records = collect_models_metadata(models_root)
    write_manifest(records, output_manifest)
    for r in sorted(records, key=lambda x: x.get("symbol") or ""):
        print(
            f"{r.get('symbol')}: price={r.get('price_model_name')} mse={r.get('price_test_mse')} mae={r.get('price_test_mae')} "
            f"dir={r.get('direction_model_name')} acc={r.get('direction_test_accuracy')} features={r.get('dataset_features')}"
        )
    print(str(output_manifest))


if __name__ == "__main__":
    main()
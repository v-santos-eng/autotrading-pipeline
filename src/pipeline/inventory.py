from pathlib import Path
import json


def collect_models_metadata(models_root: Path):
    rows = []
    if not models_root.is_dir():
        return rows  # nothing trained yet
    for symbol_dir in sorted(models_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        meta_path = symbol_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "symbol": data.get("symbol"),
            "price_model_name": data.get("price_model_name"),
            "direction_model_name": data.get("direction_model_name"),
            "price_test_mse": (data.get("price_performance") or {}).get("test_mse"),
            "price_test_mae": (data.get("price_performance") or {}).get("test_mae"),
            "direction_test_accuracy": (data.get("direction_performance") or {}).get("test_accuracy"),
            "trained_at": data.get("trained_at"),
            "dataset_train_size": (data.get("dataset_info") or {}).get("train_size"),
            "dataset_test_size": (data.get("dataset_info") or {}).get("test_size"),
            "dataset_features": (data.get("dataset_info") or {}).get("features"),
        })
    return rows


def write_manifest(records, output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump({"inventory": records}, f)

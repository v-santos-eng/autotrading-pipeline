from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelsInventoryConfig:
    models_root: Path
    output_manifest: Path
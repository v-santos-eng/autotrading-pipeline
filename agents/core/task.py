from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Task:
    id: Optional[str]
    description: str
    params: Dict[str, Any]

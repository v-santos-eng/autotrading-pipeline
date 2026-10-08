from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .task import Task


class Agent(ABC):
    name: str
    capabilities: List[str]

    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities

    @abstractmethod
    def run(self, task: Task) -> Dict[str, Any]:
        ...
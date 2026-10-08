from typing import Any, Dict

from ..core.agent import Agent
from ..core.task import Task
from ..registry import register_agent


class EchoAgent(Agent):
    def __init__(self):
        super().__init__(name="echo", capabilities=["echo", "debug"])

    def run(self, task: Task) -> Dict[str, Any]:
        return {"echo": task.description, "params": task.params}


register_agent(EchoAgent())
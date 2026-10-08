from typing import Any, Dict

from agents.core.task import Task


class EchoAgent:
    name = 'echo'
    description = 'Echoes the task description'

    def run(self, task: Task) -> Dict[str, Any]:
        return { 'echo': task.description }
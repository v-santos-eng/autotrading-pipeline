from typing import Any, Dict, List

from ..core.agent import Agent
from ..core.task import Task
from ..registry import register_agent


class SearchAgent(Agent):
    def __init__(self):
        super().__init__(name="search", capabilities=["search", "survey"])

    def run(self, task: Task) -> Dict[str, Any]:
        query = task.description
        results: List[Dict[str, Any]] = [
            {"title": "Result A", "snippet": f"Found related to {query}", "url": "http://example.com/a"},
            {"title": "Result B", "snippet": f"Also relevant: {query}", "url": "http://example.com/b"},
        ]
        return {"query": query, "results": results}


register_agent(SearchAgent())
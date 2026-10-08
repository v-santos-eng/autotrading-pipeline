from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class Task:
    id: str
    description: str
    params: Dict[str, Any]


@dataclass
class Result:
    ok: bool
    summary: str
    data: Any = None


class Agent(Protocol):
    name: str
    description: str
    capabilities: List[str]

    def run(self, task: Task) -> Result:
        ...


class Registry:
    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def list(self) -> List[Agent]:
        return list(self._agents.values())


class Runner:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def run(self, agent_name: str, task: Task) -> Result:
        agent = self._registry.get(agent_name)
        if not agent:
            return Result(False, f"Agent not found: {agent_name}")
        try:
            return agent.run(task)
        except Exception as e:
            return Result(False, str(e))
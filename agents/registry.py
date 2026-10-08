from typing import Dict, Optional

from agents.agents.echo import EchoAgent


_REGISTRY: Dict[str, object] = {
    EchoAgent.name: EchoAgent(),
}


def get_agent(name: str) -> Optional[object]:
    return _REGISTRY.get(name)


def list_agents() -> Dict[str, str]:
    return { 'echo': EchoAgent.description }

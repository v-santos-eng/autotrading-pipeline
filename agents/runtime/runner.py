from typing import Any, Dict

from agents.core.task import Task
from agents.registry import get_agent


class Runner:
    def run_task(self, agent_name: str, task: Task) -> Dict[str, Any]:
        agent = get_agent(agent_name)
        if agent is None:
            return {
                'status': 'failed',
                'error': f'Agent not found: {agent_name}'
            }
        result = agent.run(task)
        return {
            'status': 'completed',
            'result': result
        }

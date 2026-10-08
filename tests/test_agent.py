from agents.core.task import Task
from agents.registry import get_agent


def test_echo_agent():
    agent = get_agent("echo")
    assert agent is not None
    t = Task(id="t1", description="hello", params={})
    result = agent.run(t)
    assert result["echo"] == "hello"
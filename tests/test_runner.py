from agents.core.task import Task
from agents.runtime.runner import Runner


def test_runner_routes_to_echo():
    r = Runner()
    t = Task(id="t2", description="world", params={})
    out = r.run_task("echo", t)
    assert out["status"] == "completed"
    assert out["result"]["echo"] == "world"
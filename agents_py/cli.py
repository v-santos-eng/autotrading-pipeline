import argparse
import json
import time

from agents_py.core import Registry, Runner, Task
from agents_py.agents import SearchAgent, CodeSurveyAgent, PlanAgent


def make_registry() -> Registry:
    r = Registry()
    r.register(SearchAgent())
    r.register(CodeSurveyAgent())
    r.register(PlanAgent())
    return r


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents", description="Run and manage local agents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    runp = sub.add_parser("run")
    runp.add_argument("--agent", "-a", required=True)
    runp.add_argument("--task", "-t", required=True)
    runp.add_argument("--params", "-p")

    args = parser.parse_args()
    registry = make_registry()
    runner = Runner(registry)

    if args.cmd == "list":
        for a in registry.list():
            print(f"{a.name} - {a.description}")
        return

    if args.cmd == "run":
        params = json.loads(args.params) if args.params else {}
        task = Task(id=str(int(time.time() * 1000)), description=args.task, params=params)
        result = runner.run(args.agent, task)
        print(json.dumps({"result": result.__dict__}, indent=2))


if __name__ == "__main__":
    main()

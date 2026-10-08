import json
from typing import Any, Dict

import typer

from agents.core.task import Task
from agents.registry import list_agents
from agents.runtime.runner import Runner


app = typer.Typer(help='Agents CLI')


@app.command()
def list():
    for name, desc in list_agents().items():
        typer.echo(f"{name}: {desc}")


@app.command()
def run(agent: str, description: str, params: str = typer.Option('{}', '--params', help='JSON params')):
    try:
        p: Dict[str, Any] = json.loads(params)
    except Exception as e:
        raise typer.BadParameter(f'Invalid JSON for params: {e}')
    t = Task(id=None, description=description, params=p)
    out = Runner().run_task(agent, t)
    typer.echo(json.dumps(out['result'], indent=2))


if __name__ == '__main__':
    app()

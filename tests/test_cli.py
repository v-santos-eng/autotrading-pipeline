from typer.testing import CliRunner

from agents.cli import app


runner = CliRunner()


def test_cli_list():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "echo:" in result.stdout


def test_cli_run_echo():
    result = runner.invoke(app, ["run", "echo", "hi", "--params", "{}"])
    assert result.exit_code == 0
    assert "\"echo\": \"hi\"" in result.stdout
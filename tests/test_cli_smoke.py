from typer.testing import CliRunner

from nanorouter.cli import app

runner = CliRunner()


def test_data_command(tmp_path):
    result = runner.invoke(app, ["data", "--out", str(tmp_path), "--n", "120"])
    assert result.exit_code == 0
    assert (tmp_path / "train.jsonl").exists()

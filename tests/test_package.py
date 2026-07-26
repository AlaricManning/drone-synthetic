import subprocess
import sys

import dronesynth


def test_version():
    assert dronesynth.__version__


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "dronesynth.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for command in ("ingest", "convert", "build", "submit"):
        assert command in result.stdout


def test_cli_reports_a_refusal_without_a_traceback(tmp_path):
    """Refusing is the build's job, so it should read as an error, not a crash."""
    config = tmp_path / "build.yaml"
    config.write_text("runs: []\n")

    result = subprocess.run(
        [sys.executable, "-m", "dronesynth.cli", "build", "--config", str(config),
         "--version", "v002"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr

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
    for command in ("ingest", "convert", "submit"):
        assert command in result.stdout

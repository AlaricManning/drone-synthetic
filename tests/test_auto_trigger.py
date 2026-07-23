import importlib.util
from pathlib import Path

import pytest

_SOURCE = Path(__file__).parent.parent / "infra" / "lambda" / "auto_trigger.py"
spec = importlib.util.spec_from_file_location("auto_trigger", _SOURCE)
auto_trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_trigger)


class StubBatchClient:
    def __init__(self):
        self.calls = []

    def submit_job(self, **kwargs):
        self.calls.append(kwargs)
        return {"jobId": "job-42"}


def test_run_id_from_key():
    assert auto_trigger.run_id_from_key("raw/run_0002/manifest.json") == "run_0002"


def test_run_id_from_unexpected_keys_rejected():
    for key in (
        "raw/run_0002/normal/frame_000000.png",  # not a manifest
        "datasets/v001/yolo/dataset.yaml",       # wrong prefix
        "raw/manifest.json",                     # no run segment
    ):
        with pytest.raises(ValueError, match="unexpected key"):
            auto_trigger.run_id_from_key(key)


def test_handler_submits_auto_versioned_job(monkeypatch):
    stub = StubBatchClient()
    monkeypatch.setattr(auto_trigger, "_batch", stub)
    monkeypatch.setenv("JOB_QUEUE", "dronesynth-convert")
    monkeypatch.setenv("JOB_DEFINITION", "dronesynth-convert")

    event = {"detail": {"object": {"key": "raw/run_0002/manifest.json"}}}
    result = auto_trigger.handler(event, context=None)

    assert result == {"jobId": "job-42", "runId": "run_0002", "version": "auto-run_0002"}
    (call,) = stub.calls
    assert call["jobName"] == "convert-run_0002-auto-run_0002"
    assert call["containerOverrides"]["command"] == [
        "--run-id", "run_0002", "--version", "auto-run_0002",
    ]

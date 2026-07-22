from dronesynth.batch import submit_conversion


class StubBatchClient:
    def __init__(self):
        self.calls = []

    def submit_job(self, **kwargs):
        self.calls.append(kwargs)
        return {"jobId": "abc-123"}


def test_submit_conversion_builds_the_right_job():
    client = StubBatchClient()
    job = submit_conversion("run_0001", "v001", client=client)

    assert job.job_id == "abc-123"
    assert job.job_name == "convert-run_0001-v001"
    assert job.queue == "dronesynth-convert"

    (call,) = client.calls
    assert call["jobName"] == "convert-run_0001-v001"
    assert call["jobQueue"] == "dronesynth-convert"
    assert call["jobDefinition"] == "dronesynth-convert"
    assert call["containerOverrides"]["command"] == [
        "--run-id", "run_0001", "--version", "v001",
    ]


def test_submit_conversion_custom_queue():
    client = StubBatchClient()
    job = submit_conversion(
        "run_0002", "v002", queue="other-queue", job_definition="other-def", client=client
    )
    assert job.queue == "other-queue"
    assert client.calls[0]["jobDefinition"] == "other-def"

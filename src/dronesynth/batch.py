"""Submit conversion jobs to AWS Batch.

Submission is thin on purpose: the job definition (image, roles, resources)
lives in Terraform, and the container's entrypoint bakes in the S3 config —
all a submission contributes is *which run* and *which dataset version*,
passed as the container command.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_QUEUE = "dronesynth-convert"
DEFAULT_JOB_DEFINITION = "dronesynth-convert"


@dataclass(frozen=True)
class SubmittedJob:
    job_id: str
    job_name: str
    queue: str


def submit_conversion(
    run_id: str,
    dataset_version: str,
    *,
    queue: str = DEFAULT_QUEUE,
    job_definition: str = DEFAULT_JOB_DEFINITION,
    client=None,
) -> SubmittedJob:
    if client is None:
        import boto3

        client = boto3.client("batch")

    job_name = f"convert-{run_id}-{dataset_version}"
    response = client.submit_job(
        jobName=job_name,
        jobQueue=queue,
        jobDefinition=job_definition,
        containerOverrides={
            "command": ["--run-id", run_id, "--version", dataset_version],
        },
    )
    return SubmittedJob(job_id=response["jobId"], job_name=job_name, queue=queue)

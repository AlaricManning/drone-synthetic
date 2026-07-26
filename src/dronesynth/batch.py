"""Submit conversion jobs to AWS Batch.

Submission is thin on purpose: the job definition (image, roles, resources,
subcommand and config) lives in Terraform, and all a submission contributes is
*which run* and *which dataset version*.

Those two arrive as job parameters filling Ref:: placeholders, rather than as a
container command override. An override replaces the command outright, so
passing a run id that way would also erase the subcommand and config path the
job definition supplies — which is how the image came to have `convert` baked
into its entrypoint, and why nothing else could run from it.
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
        parameters={"run_id": run_id, "version": dataset_version},
    )
    return SubmittedJob(job_id=response["jobId"], job_name=job_name, queue=queue)

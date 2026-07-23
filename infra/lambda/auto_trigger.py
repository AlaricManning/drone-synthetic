"""EventBridge -> Batch: convert a run the moment its manifest lands.

Ingest writes the manifest last, so an "Object Created" event for
``raw/<run_id>/manifest.json`` means the run is complete by construction —
the only trigger-safe moment. Auto-triggered conversions write dataset
version ``auto-<run_id>``: stateless, deterministic, and impossible to
collide because runs are immutable. Curated multi-run versions (v00N) stay
human-submitted.

This file is deployed as-is by Terraform (see auto-trigger.tf); it has no
dependency on the dronesynth package — boto3 ships with the Lambda runtime.
"""

import os

import boto3

_batch = None


def _client():
    global _batch
    if _batch is None:
        _batch = boto3.client("batch")
    return _batch


def run_id_from_key(key: str) -> str:
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != "raw" or parts[2] != "manifest.json":
        raise ValueError(f"unexpected key for auto-trigger: {key!r}")
    return parts[1]


def handler(event, context):
    run_id = run_id_from_key(event["detail"]["object"]["key"])
    version = f"auto-{run_id}"
    response = _client().submit_job(
        jobName=f"convert-{run_id}-{version}",
        jobQueue=os.environ["JOB_QUEUE"],
        jobDefinition=os.environ["JOB_DEFINITION"],
        containerOverrides={"command": ["--run-id", run_id, "--version", version]},
    )
    print(f"submitted {response['jobId']}: {run_id} -> {version}")
    return {"jobId": response["jobId"], "runId": run_id, "version": version}

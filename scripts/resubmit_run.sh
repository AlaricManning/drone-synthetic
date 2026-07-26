#!/bin/bash
# Resubmit the auto-conversion for one run, matching exactly what the Lambda
# would have sent. For recovering runs whose Batch job died on something
# transient (ECR pull timeouts, capacity) rather than on the data.
set -eu

REGION=${REGION:-us-east-1}
QUEUE=${QUEUE:-dronesynth-convert}
JOB_DEF=${JOB_DEF:-dronesynth-convert}

run_id=${1:?usage: resubmit_run.sh <run-id>}
version="auto-$run_id"

job_id=$(aws batch submit-job \
  --job-name "convert-$run_id-retry" \
  --job-queue "$QUEUE" \
  --job-definition "$JOB_DEF" \
  --region "$REGION" \
  --container-overrides "command=[--run-id,$run_id,--version,$version]" \
  --query jobId --output text)

echo "submitted $job_id: $run_id -> $version"

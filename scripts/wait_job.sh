#!/bin/bash
# Poll a Batch job until it settles, then print why it ended.
set -eu

REGION=${REGION:-us-east-1}
job_id=${1:?usage: wait_job.sh <job-id>}
tries=${2:-40}

for i in $(seq 1 "$tries"); do
  status=$(aws batch describe-jobs --jobs "$job_id" --region "$REGION" \
    --query 'jobs[0].status' --output text)
  echo "[$i] $status"
  case "$status" in
    SUCCEEDED|FAILED) break ;;
  esac
  sleep 20
done

aws batch describe-jobs --jobs "$job_id" --region "$REGION" \
  --query 'jobs[0].{status:status,reason:statusReason,exit:container.exitCode}' \
  --output json

#!/bin/bash
# Remove runs from the bucket entirely: raw frames, the auto-converted dataset
# version the Lambda produced for them, and their QC output. Takes a file of
# run IDs, one per line.
#
# The bucket is versioned, so these deletes leave delete markers rather than
# purging bytes; the runs stop being visible to the dataset and to any future
# curated version, which is what matters here.
#
# Listing this bucket costs about a minute per call, so we list it once up
# front and resolve every run against that local snapshot rather than probing
# each prefix. Deletes go out in batches of 1000, the delete-objects maximum.
#
#   scripts/prune_runs.sh <run-id-file>          # dry run, prints what would go
#   scripts/prune_runs.sh <run-id-file> --apply  # actually delete
set -euo pipefail

BUCKET=${BUCKET:-drone-synthetic-am}
REGION=${REGION:-us-east-1}

run_file=${1:?usage: prune_runs.sh <run-id-file> [--apply]}
apply=${2:-}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

snapshot=${SNAPSHOT:-$work/keys.txt}
if [[ -s "$snapshot" ]]; then
  echo "reusing key snapshot at $snapshot ($(wc -l < "$snapshot") keys)"
else
  echo "listing s3://$BUCKET (one pass, takes a minute)..."
  aws s3 ls "s3://$BUCKET/" --recursive --region "$REGION" \
    | awk '{ $1=""; $2=""; $3=""; sub(/^ +/, ""); print }' > "$snapshot"
  echo "  $(wc -l < "$snapshot") keys total"
fi

# Every key a run owns lives under one of three prefixes, so one anchored
# pattern per run is enough to find all of them.
: > "$work/patterns.txt"
while read -r run_id; do
  [[ -z "$run_id" ]] && continue
  printf '^raw/%s/\n^datasets/auto-%s/\n^qc/%s/\n' "$run_id" "$run_id" "$run_id" >> "$work/patterns.txt"
done < "$run_file"

grep -E -f "$work/patterns.txt" "$snapshot" | sort -u > "$work/doomed.txt" || true
count=$(wc -l < "$work/doomed.txt")

echo
echo "runs to prune:    $(grep -cve '^$' "$run_file")"
echo "objects matched:  $count"
echo
echo "per-prefix breakdown:"
sed -E 's#^(raw|qc)/([^/]+)/.*#\1/\2#; s#^(datasets/auto-[^/]+)/.*#\1#' "$work/doomed.txt" \
  | sort | uniq -c | sed 's/^/  /'

if [[ "$count" -eq 0 ]]; then
  echo
  echo "nothing to do"
  exit 0
fi

if [[ "$apply" != "--apply" ]]; then
  echo
  echo "DRY RUN — pass --apply to delete these $count objects"
  exit 0
fi

echo
echo "deleting $count objects from s3://$BUCKET..."
split -l 1000 "$work/doomed.txt" "$work/batch."
deleted=0
for batch in "$work"/batch.*; do
  python3 -c '
import json, sys
keys = [line.rstrip("\n") for line in open(sys.argv[1]) if line.strip()]
json.dump({"Objects": [{"Key": k} for k in keys], "Quiet": True}, open(sys.argv[2], "w"))
' "$batch" "$batch.json"
  aws s3api delete-objects \
    --bucket "$BUCKET" --region "$REGION" \
    --delete "file://$batch.json" > /dev/null
  deleted=$((deleted + $(wc -l < "$batch")))
  echo "  $deleted / $count"
done

echo "done — $deleted objects deleted"

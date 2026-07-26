#!/bin/bash
# One full listing of the bucket, cached to /tmp/s3_keys.txt. Listing this
# bucket is slow enough that anything needing key-level detail should read the
# snapshot instead of calling S3 again.
set -eu

BUCKET=${BUCKET:-drone-synthetic-am}
REGION=${REGION:-us-east-1}
OUT=${OUT:-/tmp/s3_keys.txt}

echo "listing s3://$BUCKET ..."
time aws s3 ls "s3://$BUCKET/" --recursive --region "$REGION" \
  | awk '{ $1=""; $2=""; $3=""; sub(/^ +/, ""); print }' > "$OUT"

echo
echo "keys: $(wc -l < "$OUT")  ->  $OUT"
echo
echo "--- top-level prefixes ---"
cut -d/ -f1 "$OUT" | sort | uniq -c
echo
echo "--- committed runs (manifest present) ---"
grep -c '^raw/.*/manifest\.json$' "$OUT"

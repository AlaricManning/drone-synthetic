#!/bin/bash
# End-to-end check on a set of runs: raw committed, auto-conversion published,
# QC report written, and the QC numbers themselves sane. Presence checks read
# the key snapshot rather than probing each prefix -- one listing of the whole
# bucket costs seconds, while a per-prefix call costs about a minute.
#
#   snapshot_keys.sh                       # refresh the snapshot first
#   verify_corpus.sh <run-id-file>
set -eu

BUCKET=${BUCKET:-drone-synthetic-am}
REGION=${REGION:-us-east-1}
SNAPSHOT=${SNAPSHOT:-/tmp/s3_keys.txt}
QC_DIR=${QC_DIR:-/tmp/qc_reports}
# Frames per clip, which fixes how many objects a complete run owns:
#   raw      frames rgb + frames mask + manifest.json + job.json
#   dataset  annotations json + provenance sidecar
#   qc       frames debug renders + report.json
#
# A conversion is two objects under datasets/ regardless of frame count: it
# writes annotations and provenance and nothing else. Runs converted before the
# per-run YOLO export was dropped also carry frames*2 + 1 export objects, so a
# mixed set reports mismatches for the older ones -- expected, and the reason
# DS_EXTRA exists rather than a second threshold to remember.
FRAMES=${FRAMES:-60}
# Set to the export size to check runs converted before the export was dropped:
#   DS_EXTRA=$((60 * 2 + 1)) verify_corpus.sh runs.txt
DS_EXTRA=${DS_EXTRA:-0}

run_file=${1:?usage: verify_corpus.sh <run-id-file>}

want_raw=$((FRAMES * 2 + 2))
want_ds=$((2 + DS_EXTRA))
want_qc=$((FRAMES + 1))

echo "=== presence: raw / dataset / qc (want $want_raw/$want_ds/$want_qc) ==="
bad=0
while read -r r; do
  [[ -z "$r" ]] && continue
  raw=$(grep -c "^raw/$r/" "$SNAPSHOT" || true)
  ds=$(grep -c "^datasets/auto-$r/" "$SNAPSHOT" || true)
  qc=$(grep -c "^qc/$r/" "$SNAPSHOT" || true)
  if [[ "$raw" -ne "$want_raw" || "$ds" -ne "$want_ds" || "$qc" -ne "$want_qc" ]]; then
    echo "  MISMATCH $r  raw=$raw dataset=$ds qc=$qc"
    bad=$((bad + 1))
  fi
done < "$run_file"
echo "  runs with unexpected object counts: $bad"
echo

echo "=== pulling QC reports ==="
mkdir -p "$QC_DIR"
aws s3 cp "s3://$BUCKET/qc/" "$QC_DIR" --recursive --region "$REGION" \
  --exclude "*" --include "*/report.json" --only-show-errors
echo "  reports on disk: $(find "$QC_DIR" -name report.json | wc -l)"
echo

python3 - "$run_file" "$QC_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_file, qc_dir = Path(sys.argv[1]), Path(sys.argv[2])
runs = [line.strip() for line in run_file.read_text().splitlines() if line.strip()]

missing = []
frames = boxes = empty = 0
areas, fills = [], []
# Flag reasons carry frame-specific numbers, so bucket them by shape to see
# which kinds occur rather than 3000 near-identical strings.
shapes = Counter()
fragmented = []

for run in runs:
    report = qc_dir / run / "report.json"
    if not report.is_file():
        missing.append(run)
        continue
    data = json.loads(report.read_text())
    frames += data["frames"]
    boxes += data["total_boxes"]
    empty += data["empty_frames"]
    if data["box_area_min"] is not None:
        areas += [data["box_area_min"], data["box_area_max"]]
        fills += [data["fill_ratio_min"], data["fill_ratio_max"]]
    for flag in data["flags"]:
        reason = flag["reason"]
        shapes["".join("N" if c.isdigit() else c for c in reason)] += 1
        if "pieces" in reason:
            fragmented.append((run, flag["frame_index"], reason))

print("=== QC summary ===")
print(f"  runs checked:   {len(runs) - len(missing)} / {len(runs)}")
print(f"  total frames:   {frames}")
print(f"  total boxes:    {boxes}")
print(f"  empty frames:   {empty}")
if areas:
    print(f"  box area:       {min(areas)} .. {max(areas)} px^2")
    print(f"  fill ratio:     {min(fills):.3f} .. {max(fills):.3f}")
if missing:
    print(f"  MISSING REPORTS ({len(missing)}): {', '.join(missing)}")

print()
print("=== QC flags by reason ===")
if not shapes:
    print("  none")
for shape, count in shapes.most_common():
    print(f"  {count:5d}  {shape}")

if fragmented:
    print()
    print(f"=== fragmented masks: {len(fragmented)} frame(s) ===")
    for run, index, reason in fragmented:
        print(f"  {run} frame {index}: {reason}")

sys.exit(1 if missing else 0)
PY

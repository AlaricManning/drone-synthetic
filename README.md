# drone-synthetic

Synthetic training-data pipeline for drone detection. UE 5.5 + EasySynth render
paired frames — a normal render and a drone-on-black mask render from identical
camera paths — and this pipeline turns those pairs into versioned, QC'd YOLO
datasets. S3 is the system of record; conversion runs as a containerized AWS
Batch job.

## Architecture

```
Windows (UE 5.5 + EasySynth)
┌──────────────────────────────────────────┐
│ render → local disk (scratch)            │  renders are messy and can fail;
└────────────────┬─────────────────────────┘  local disk absorbs that
                 │
                 │  dronesynth ingest   (validates pairing/frame counts, writes
                 │                       manifest, uploads frames first,
                 ▼                       manifest LAST)
        s3://<bucket>/raw/<run_id>/
            ├── normal/
            ├── mask/                     a run with a manifest is complete by
            └── manifest.json             construction; without one, ignore it
                 │
                 │  dronesynth submit --run <run_id>
                 ▼
        AWS Batch job (Fargate, CPU) — containerized converter
            mask threshold → boxes → canonical JSON → YOLO export → QC
                 │
                 ▼
        s3://<bucket>/datasets/<version>/   canonical per-frame annotations
            ├── annotations/                 + YOLO images/labels layout
            └── yolo/
        s3://<bucket>/qc/<run_id>/          QC report + debug box renders
```

## Design decisions

- **Runs are the atomic unit.** Each capture session is one immutable
  `run_id` with a manifest recording UE map, drone model, camera path, capture
  date, and (later) domain-randomization parameters and seed. Runs are the
  unit of ingest, QC, provenance, and train/val splitting.
- **Canonical JSON annotations; YOLO is an export.** Mask renders carry
  segmentation information for free. Conversion writes per-frame JSON
  (boxes, mask area, fill ratio) as the source of truth and generates the
  YOLO layout from it, so future COCO or segmentation exports are new
  exporters, not rewrites.
- **Datasets are versioned and deterministic.** A dataset version is fully
  determined by (input runs, conversion config). Same inputs, same output,
  always re-derivable.
- **Splits are run-level, never frame-level.** Consecutive frames from one
  camera path are near-duplicates; frame-level random splits leak train into
  val and inflate metrics. Whole runs are held out for validation.
- **QC is the proof of quality.** Nothing downstream trains on this data
  within the pipeline, so the QC report (boxes per frame, box size
  distribution, mask fill ratio, empty-frame counts, flagged outliers) and
  debug renders are the evidence the labels are good.

## Security model

| Identity       | Permissions                                  | Used by                 |
|----------------|----------------------------------------------|-------------------------|
| ingest user    | put-only on `raw/*` — no list, no delete     | `dronesynth ingest`     |
| batch job role | read `raw/*`; write `datasets/*` and `qc/*`  | the Fargate conversion job |
| admin          | full                                         | Terraform applies, browsing |

Leaked ingest credentials must not allow enumerating or deleting captured
data. No credentials live in this repo, tracked or otherwise. All AWS
resources are provisioned by Terraform in `infra/`.

## Repository layout

```
configs/               conversion config: mask threshold, split policy,
                       class map, storage roots (local paths or s3:// URIs)
src/dronesynth/
  ingest/              run registration, validation, manifest, S3 sync
  datagen/             pairing, mask→box, canonical JSON, exporters, QC
  storage/             local/S3 abstraction — same code both sides
  cli.py               ingest / convert / submit entrypoints
infra/                 Terraform: bucket, IAM, ECR, Batch
tests/
data/                  gitignored local staging (raw/, datasets/, qc/)
```

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
dronesynth --help
pytest
```

Development happens in WSL; EasySynth captures on the Windows side are read
via `/mnt/c/datasets` during local development and via S3 in production.

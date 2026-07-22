# drone-synthetic

Synthetic training-data pipeline for drone detection. UE 5.5 + EasySynth render
paired frames — a normal render and a drone-on-black mask render from identical
camera paths — and this pipeline turns those pairs into versioned, QC'd YOLO
datasets. The production shape it is being built toward: S3 as the system of
record, with conversion running as a containerized AWS Batch job.

## Architecture

The diagram shows the target architecture. The S3 system of record is live,
ingest writes to it, and the containerized converter runs the full S3-to-S3
path (proven locally via `docker run`) — AWS Batch and `dronesynth submit`
are the parts still to come.

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

## Where this fits in the larger system

This repo is one station of a detection data flywheel, alongside
[object-tracker](https://github.com/AlaricManning/object-tracker) and
[object-tracker-pipeline](https://github.com/AlaricManning/object-tracker-pipeline).
COCO has no `drone` class, so the tracker's stock YOLOv11 cannot detect
drones — custom-trained weights are required, and this pipeline manufactures
the labeled data that trains them:

```
┌─ drone-synthetic (this repo) ──┐
│ UE renders → runs → datasets/vN ──► training (external for now)
└────────────────────────────────┘        │ drone-capable YOLO weights
                                          ▼
┌─ object-tracker (edge) ────────────────────────────┐
│ custom weights replace stock yolo11n; "drone"      │
│ becomes the target class → clips + KLV → s3 raw/   │
└────────────────────────┬───────────────────────────┘
                         ▼
┌─ object-tracker-pipeline ──────────────────────────┐
│ KLV → Parquet catalog → Athena/DuckDB              │
└────────────────────────┬───────────────────────────┘
                         │  query: where does the model struggle?
                         ▼
        findings → domain-randomization params in the
        next runs' manifests → new datasets → retrain ──┐
                         ▲──────────────────────────────┘
```

The tracker's `near_misses` confidence tier is ready-made hard-example
mining: queries over real detections show where the model is unsure, and
those findings become domain-randomization parameters in the next capture
runs' manifests (the reserved `randomization`/`seed` fields are the return
path). Real clips from the field are also the only true eval set — the
synthetic run-level val split measures sim-to-sim generalization only.

The systems couple through artifacts, never code: dataset version → model
weights → KLV catalog → randomization params. Separate buckets, separate
IAM identities.

## Design decisions

- **Runs are the atomic unit.** Each capture session is one immutable
  `run_id` with a manifest recording UE map, drone model, camera path, capture
  date, and (later) domain-randomization parameters and seed. Runs are the
  unit of ingest, QC, provenance, and train/val splitting. Ingest writes the
  manifest only after every frame is in place, so a run without a manifest is
  always debris from a failed ingest, never a real run — and a run with one
  is immutable: re-ingesting an existing id is an error.
- **Canonical JSON annotations; YOLO is an export.** Mask renders carry
  segmentation information for free. Conversion writes per-frame JSON
  (boxes, mask area, fill ratio) as the source of truth and generates the
  YOLO layout from it, so future COCO or segmentation exports are new
  exporters, not rewrites.
- **Datasets are versioned and deterministic.** A dataset version is fully
  determined by (input runs, conversion config). Same inputs, same output,
  always re-derivable.
- **The train/val split ships with the dataset, run-level, never
  frame-level.** Consecutive frames from one camera path are near-duplicate
  images, so a random frame-level split leaks train data into val and
  inflates metrics — and a consumer handed loose frames can't know that,
  since the sequence structure isn't visible in a folder of PNGs. As with
  COCO and ImageNet, the producer defines the split: whole runs are held
  out via `split.val_runs` in the config, and frame-level splitting is
  rejected outright. (With one run captured, val is empty until the first
  held-out run is added.)
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

## Usage

After each render session, register the capture as a run:

```bash
AWS_PROFILE=drone-synth-ingest dronesynth ingest --config configs/convert.yaml \
  --normal /mnt/c/datasets/drone_normal --mask /mnt/c/datasets/drone_mask \
  --run-id run_0001 --ue-map testLevel --drone-model White_Drone \
  --raw-root s3://drone-synthetic-am/raw
```

This validates the capture (strict normal/mask pairing — broken renders are
rejected before anything is uploaded), flattens it into
`raw/run_0001/{normal,mask}/`, and writes the manifest last — with if-absent
semantics, so an existing run can never be overwritten. Omit `--raw-root`
(and the profile) to ingest to the local `data/raw` staging area from config
instead; local runs are what `convert` reads until the Batch job lands.

Then convert a registered run into a versioned dataset:

```bash
dronesynth convert --config configs/convert.yaml --run-id run_0001 --version v001
```

This writes canonical annotations and the YOLO layout to
`data/datasets/v001/` and the QC report plus debug renders (frames with the
detected boxes drawn on) to `data/qc/run_0001/`. Review flagged frames — and
ideally scrub the debug folder — before treating the dataset as good.

To run the conversion as the container does in production — everything in
and out of S3, using `configs/convert.s3.yaml`:

```bash
docker build -f docker/Dockerfile -t dronesynth-convert .
docker run --rm -v ~/.aws:/home/app/.aws:ro -e AWS_PROFILE=default \
  dronesynth-convert --run-id run_0001 --version v001
```

Credentials are never baked into the image: locally they come from the
read-only `~/.aws` mount; on Batch they will come from the job role.

## Infrastructure

AWS resources are managed by Terraform in `infra/` and applied manually with
admin credentials:

```bash
cd infra
terraform init
terraform apply -var bucket_name=drone-synthetic-am
```

The ingest access key is created outside Terraform (state files store
secrets in plaintext) and lives only in `~/.aws/credentials`:

```bash
aws iam create-access-key --user-name drone-synth-ingest
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

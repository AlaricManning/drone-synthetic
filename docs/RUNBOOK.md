# Runbook

Operator procedures for the drone-synthetic pipeline: what to type, what to
expect, and what to do when a step fails. For what the system *is*, read the
[README](../README.md).

Assumed one-time setup (already done on the current machine): the repo's
`.venv` with `pip install -e ".[dev]"`, AWS profiles `default` (admin) and
`drone-synth-ingest` (put-only) in `~/.aws/credentials`, terraform applied
in `infra/`, docker installed, and the conversion image pushed to ECR.

---

## Render day: capture → dataset

### 1. Render the capture in UE 5.5 + EasySynth

Two rendering passes from the **identical camera path**: the normal render
and the drone-on-black mask render. Save to separate folders, e.g.
`C:\datasets\drone_normal` and `C:\datasets\drone_mask`. Nesting like
`CameraComponent/ColorImage/` is fine — ingest flattens it. Frame numbers
must be the trailing digits of each filename (EasySynth default).

### 2. Pick the next run id

Runs are immutable — every capture gets a fresh id (`run_0002`, `run_0003`,
…). To see what exists (admin profile — the ingest profile cannot list):

```bash
aws s3 ls s3://drone-synthetic-am/raw/
```

### 3. Ingest

```bash
AWS_PROFILE=drone-synth-ingest dronesynth ingest --config configs/convert.yaml \
  --normal /mnt/c/datasets/drone_normal --mask /mnt/c/datasets/drone_mask \
  --run-id run_0002 --ue-map <UE level name> --drone-model <drone asset name> \
  --raw-root s3://drone-synthetic-am/raw
```

Add `--captured-at YYYY-MM-DD` if the render wasn't today. Expected output:

```
registered run_0002: <N> frame pairs
  sequence: <sequence name> (<map>, <model>)
  location: s3://drone-synthetic-am/raw/run_0002
```

Upload takes a few minutes per few hundred MB. The manifest is written
last, so an interrupted ingest never produces a valid-looking run — just
run the same command again.

### 4. Submit the conversion

```bash
dronesynth submit --run-id run_0002 --version v002
```

Pick the version: re-converting existing runs with unchanged config can
reuse the version (outputs are deterministic and overwrite in place); new
runs or changed config mean a new version. Expected output is the job id
plus a ready-made status command.

### 5. Watch the job

```bash
aws batch describe-jobs --jobs <job-id> --query 'jobs[0].{status:status,reason:statusReason}'
```

Lifecycle: `SUBMITTED → RUNNABLE → STARTING → RUNNING → SUCCEEDED`, with a
minute or two before RUNNING while Fargate provisions and pulls the image.
The container's own summary lands in CloudWatch:

```bash
aws logs tail /aws/batch/job --since 15m
```

Expect `run run_0002: N frames, M boxes, K empty frames … ` and ideally
`no flags`.

### 6. Review QC before trusting the dataset

```bash
aws s3 cp s3://drone-synthetic-am/qc/run_0002/report.json - | python3 -m json.tool
aws s3 sync s3://drone-synthetic-am/qc/run_0002/debug/ /tmp/qc-run_0002/
```

Open the debug renders (from Windows:
`\\wsl$\Ubuntu\tmp\qc-run_0002\`) and scrub through: every box should hug
the drone; no boxes on empty sky. If the report lists flags, eyeball those
frames first — that's what the flags are for. A dataset is not "good"
until a human has looked.

### 7. The deliverable

```
s3://drone-synthetic-am/datasets/<version>/
  annotations/<run_id>.json    canonical per-frame annotations
  yolo/                        images/, labels/, dataset.yaml
```

Consumers download the `yolo/` tree; `dataset.yaml` uses relative paths, so
it works wherever the tree is placed (ultralytics resolves `path: .`
against its configured datasets directory — point it at the download
location).

---

## Changing conversion settings (threshold, split, class map)

`configs/convert.s3.yaml` is **baked into the container image** — editing
the repo file does nothing to cloud jobs until the image is rebuilt and
pushed:

```bash
docker build -f docker/Dockerfile -t dronesynth-convert .
docker tag dronesynth-convert:latest 935961368629.dkr.ecr.us-east-1.amazonaws.com/dronesynth-convert:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 935961368629.dkr.ecr.us-east-1.amazonaws.com
docker push 935961368629.dkr.ecr.us-east-1.amazonaws.com/dronesynth-convert:latest
```

Jobs pick up `:latest` on their next start — no terraform needed. The same
procedure deploys **code** changes. Config changes that affect dataset
content should always write a **new dataset version**. Keep
`convert.yaml` (local) and `convert.s3.yaml` in lockstep on every knob
except storage roots.

To hold out runs for validation, list them under `split.val_runs` in the
config (both files), rebuild/push, and convert into a new version.

---

## When things go wrong

**Ingest says "capture is incomplete"** — the normal/mask folders disagree
(the message lists the offending frame indices). Fix the render or remove
strays, re-run ingest. Nothing was uploaded.

**Ingest says "already exists — runs are immutable"** — that run id is
taken. Use the next id. (This fires at the end on S3: the put-only
identity can't check first, so frames re-upload before the manifest write
is refused. Harmless, just slow.)

**Convert/job says "the run is corrupt" (frame count mismatch)** — stale
objects from an interrupted ingest of a different-shaped capture. Inspect
and clean with admin credentials, then re-ingest:

```bash
aws s3 ls s3://drone-synthetic-am/raw/run_0002/ --recursive
aws s3 rm s3://drone-synthetic-am/raw/run_0002/ --recursive   # removes the whole run
```

**Batch job FAILED** — `describe-jobs` shows `statusReason`; the container
log is in `aws logs tail /aws/batch/job --since 1h`. Common causes: image
missing from ECR (lifecycle expired it — rebuild and push), a config typo
in the baked config (rebuild), S3 permission errors (the job role only
reads `raw/*` and writes `datasets/*`/`qc/*` — anything else is denied by
design).

**Debugging a job without Batch** — run the identical container locally
with admin credentials:

```bash
docker run --rm -v ~/.aws:/home/app/.aws:ro -e AWS_PROFILE=default \
  dronesynth-convert --run-id run_0002 --version v002
```

Or fully local (no AWS at all): ingest without `--raw-root`, then
`dronesynth convert --config configs/convert.yaml --run-id run_0002 --version v002`
against `data/`.

**Re-running anything is safe.** Conversion is deterministic; identical
inputs overwrite identical outputs at identical keys. Runs in `raw/` are
never modified by conversion.

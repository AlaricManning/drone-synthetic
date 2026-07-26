# Plan: assembling a trainable dataset

Status: decided 2026-07-26, not yet implemented. Unlike `instance-boxes.md`
this is written before the work, so what follows is a design record and the
reasoning behind it, not a measurement. It should be frozen once built.

Second plan doc; same convention. This one exists because a request that
sounded like a config edit — hold out some runs for validation — turned out
not to be expressible in the code as written, and answering *why* turned up a
larger gap behind it.

## What is missing

The first corpus finished on 2026-07-26: 50 runs, 3000 frames, 3000 boxes,
zero empty frames. Every one of those frames converted successfully. None of
them are in a dataset anyone can train on.

Conversion is per-run and always has been. The Lambda submits one job per run
and each job writes its own dataset version, `auto-<run_id>`, holding that
run's frames and nothing else. The bucket has 69 of them. They are useful as
staging and as what QC reads, but a training set has to span runs, and nothing
assembles one.

The two curated versions that exist, `v00` and `v001`, are earlier test builds.
Each holds a single run — `run_0001`, 150 frames — and in both, every frame is
in `yolo/images/train` with `yolo/images/val` empty. The pipeline has never
produced a validation split.

## Why `split.val_runs` cannot express it

The README says whole runs are held out via `split.val_runs`. That is the
intended design and the field exists, but setting it today breaks conversion.

`convert_run` asks for an assignment covering exactly one run:

```python
assignments = split_runs([run_id], config.split.val_runs)
```

and `split_runs` rejects any named val run that is not among the runs it was
handed. With ten runs named and one being converted, the other nine are
unknown and the job fails. Naming a single run fails too, on the other 49.
**Any non-empty `split.val_runs` fails every conversion**, which is every run
the auto path touches.

The check is not wrong — `test_unknown_val_run_rejected` pins it deliberately,
and it catches a typo in a hand-written val list, which would otherwise
silently produce an empty validation set. The problem is that `split_runs` was
written for a caller that assembles many runs at once, and that caller was
never built.

## The governing requirement: audit

The decisive constraint is not convenience. It is being able to say of a set of
weights: *this model was trained on these frames, produced by this renderer
build, converted by this converter build under this config.*

Tracing that chain today, it goes dark in three places:

| Link | Status |
| --- | --- |
| Which frames | derivable — `annotations/<run_id>.json` names the runs |
| Which renderer build | recorded — `generator` in each run manifest |
| Seed and randomization | recorded in the run manifest |
| Which converter build | **nothing** — only a static `__version__` |
| Which conversion config | **nothing** — no threshold, no `min_box_area` |
| Which runs, and the split | **nothing** — no dataset-level manifest |

The sharp illustration is the mask threshold moving from 12 to 32 on
2026-07-26. That materially changed label content, and neither a dataset built
before nor one built after records which threshold produced it. It is exactly
the failure the `generator` stamp was introduced to prevent on the render side,
sitting unaddressed one stage downstream.

A consequence worth stating: the README's claim that a dataset version "is
fully determined by (input runs, conversion config)" is currently
unverifiable, because the config is not stored anywhere near the output.

### The dataset must snapshot, not reference

Pruning the 25 duplicate runs removed 7625 objects including their manifests.
A dataset that merely *referenced* run ids would have quietly lost the ability
to explain itself the moment its inputs were deleted.

So the dataset manifest embeds what it needs — each input run's id, seed,
randomization parameters, map, drone model and `generator` stamp — rather than
pointing at run manifests that may not outlive it. Inputs are mutable; the
audit record cannot depend on them.

This is also what rules out the cheaper options. An index-only dataset (text
files listing image paths in the `auto-` versions) inherits their lifetime. A
build script run on a training machine produces no artifact to attach a record
to.

## The decision

**A split is a property of an assembled dataset, not of a conversion.**

Converting one run answers "what is in these frames". Splitting answers "which
runs does this dataset train on and which does it validate against", and that
question does not exist until something decides which runs the dataset
contains. Per-run conversion has no split to make; the current code only
appears to give it one because a lone run trivially lands in train.

So: a `build` command that takes an explicit set of runs and a version name,
applies the split across them, writes one dataset, and records what it did.
`split_runs` then gets the caller it was designed for, strict validation and
all, and its typo check starts protecting something.

### Build from annotations, not by re-converting

The canonical per-frame JSON is already written, one file per run under
`auto-<run_id>/annotations/`. A build does not re-threshold a single mask: it
reads those annotations, copies the normal frames into `images/{train,val}/`,
writes label files, and emits `dataset.yaml`.

This is what "canonical JSON annotations; YOLO is an export" has always claimed
without anything depending on it. Building this way also makes a re-split cheap
— changing the hold-out re-copies files but recomputes no labels — and means a
build cannot disagree with the QC report its annotations came from.

### Complete by construction, like a run

Ingest already writes a run's manifest last, with `write_text_if_absent`, so a
run without a manifest is debris and a run with one is immutable. The build
uses the identical protocol: copy frames and labels, then write the dataset
manifest last, if-absent. A dataset version without a manifest is debris from
a failed build, and rebuilding an existing version is an error for the same
reason re-ingesting a run id is.

Reusing the primitive matters more than the elegance: it is one discipline for
both artifacts rather than two that can drift.

### What the manifest records

```
datasets/<version>/manifest.json
  version, built_at
  builder     repo, commit, dirty       -- which build assembled this
  conversion  threshold, min_box_area, class_map
  split       mode, train[], val[]
  runs[]      run_id, seed, ue_map, drone_model, randomization,
              generator, converter, frames, boxes
  totals      runs, frames, boxes, train_frames, val_frames
```

`conversion` is rolled up from the per-run converter stamps rather than read
from the build's own config, so it describes what actually produced the labels.
If the input runs were converted under differing configs or converter builds,
their labels are not comparable and the build refuses. That is a live scenario:
the six test runs predating the threshold change sit in the same bucket as the
corpus.

## Splitting

Frame-level splitting is already impossible, at three levels: config load
rejects any mode but `by_run` with the reason named inline, `split_runs` only
accepts run ids so a frame-level split is not expressible, and the exporter
looks up assignment per run so frames inherit their run's side.

**Run-level is necessary and not sufficient.** When the two render batches
collided, 25 seeds rendered twice; those pairs have different run ids and
byte-identical `trajectory` blocks. A run-level split would put one in train
and its twin in val and report no problem, because run ids differ.

The identity of a *scene* is not the run id — it is roughly (renderer commit,
config, seed), and the run id is a timestamp-based proxy that is unique per
render rather than per scene. So the build additionally refuses when a scene
identity spans the split. The cheap form is that no seed appears on both sides;
the fuller form hashes seed together with randomization, map, drone model and
generator commit, since the same seed under a different config is a different
scene.

The current corpus is clean on this — 50 runs, 50 distinct seeds, verified
after the prune — so the check guards against recurrence rather than fixing
anything outstanding.

### The hold-out

Recession and crossing clips come from different configs and are genuinely
different distributions, so the hold-out is proportional to each rather than a
flat sample of 50. At 20%: 7 of 35 recession, 3 of 15 crossing, every fifth
seed.

| | seeds |
| --- | --- |
| recession val | 1102, 1107, 1112, 1117, 1122, 1127, 1132 |
| crossing val | 2102, 2107, 2112 |

600 val frames against 2400. Seeds are independent draws, so a contiguous block
would be equally valid; spreading them guards only against ordering effects
nobody has checked for.

Worth stating plainly: 10 runs of one drone model against one sky is a thin
validation set. It measures whether the converter and renderer agree, not
whether a detector generalizes. Run-level splitting also means the effective
size is 10 scenes, not 600 samples. Both are the right things to measure now
and should not be mistaken for the other thing.

## Where it runs, and as whom

The build is storage-agnostic like everything else, so it is developed and
unit-tested against a local `data/` root with no AWS, and real builds run on
Batch.

The identity question decided this. Of the four identities, none fits a human
running a build: ingest and render are put-only on `raw/*`, and the convert job
role — which has close to the right grants — trusts only ECS tasks. That leaves
admin, for a routine repeated operation, in a repo whose security model is
explicitly least-privilege.

More practically, if local runs use admin and Batch runs use a narrow role,
"it worked locally" stops predicting "it works on Batch". This feature would
have hit that immediately: the build needs read on `datasets/*` to pull
annotations, which the convert job role does not have, so a local admin build
succeeds and the Batch build fails with `AccessDenied`.

So: **a distinct build role**, with read on `raw/*` and `datasets/*` and write
on `datasets/*` — conversion has no use for reading `datasets/*`, so sharing a
role would over-grant it. A named human principal goes in the build role's
trust policy, making local one-off builds an assume-role rather than a fifth
long-lived key: temporary credentials, identical grants in both places, and
CloudTrail attributing each build to a session.

The local path writes to the same versioned locations under the same
manifest-last protocol. A "local mode" that wrote somewhere special would undo
the point.

Server-side copy is what keeps this cheap. `Storage` has no key-to-key copy
today, only `put_file(source: Path, key)` and `get_file(key, dest: Path)`, so
a naive build round-trips roughly 9 GB through whatever runs it — over an hour
at the transfer rates observed during the corpus work. With a `copy_key`
primitive no bytes leave the bucket, and `CopyObject` needs only `GetObject` on
the source and `PutObject` on the destination, both of which the build role has.

## What lands, in order

1. `copy_key` on the `Storage` ABC and both backends.
2. Per-run converter provenance: a `provenance.json` sidecar beside the
   annotations recording converter repo, commit, dirty flag, and the mask
   config the labels were produced under. A sidecar rather than a wrapper
   because `write_annotations` emits a bare JSON list and changing its shape
   would break `read_annotations` and every existing file.
3. `dronesynth build`: build config, dataset manifest, the scene-identity
   check, manifest-last.
4. Terraform: the build role, its trust policy, and its grants.

## Deferred, deliberately

**Removing the per-run YOLO export.** Each `auto-` version holds 60 images that
already exist under `raw/`, roughly 4100 duplicated frames across 69 runs. The
storage cost is negligible — about 6 GB, cents a month — so the real argument is
that `images/train` in a per-run version does not mean "assigned to train", it
means "no split was made", which is a misleading claim encoded in a directory
name. Removing it would also cut ~90 MB of upload from every conversion job.

It is deferred because it is a breaking change to a deployed path, and bundling
it with a new build command makes both riskier. Once the build is proven, the
per-run export has no consumer and can go.

**A separate val list in the conversion config.** `split` moves to the build
config, since the split belongs to a dataset version. Relocating it costs
nothing precisely because it has never been usable.

**Aggregate QC in the build.** Per-run QC exists; batch-level properties — the
duplicate seeds, the run whose conversion silently went missing — were caught by
ad-hoc scripts. Whether the build should absorb that is open.

## Validation

Not yet done. When implemented:

- `split_runs` gets a real multi-run caller and the existing tests pass
  unchanged.
- A build over the 50-run corpus produces 2400 train and 600 val images with
  matching label counts, and no run id or seed on both sides.
- Frame counts reconcile against the QC reports the annotations came from:
  3000 images, 3000 labels, 3000 boxes.
- Re-running the same build is refused, because the manifest already exists.
- A build mixing runs converted under different thresholds is refused.
- The same build against a local root and against S3 produces identical
  manifests apart from `built_at`.

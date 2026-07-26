# Plan: assembling a trainable dataset

Status: decided and implemented 2026-07-26. Unlike `instance-boxes.md` this was written before the
work, so the body is a design record rather than a measurement; the
[Validation](#validation) section at the end records what the implementation
actually showed, including the two places the design was wrong.

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

1. `copy_from` on the `Storage` ABC and both backends. Named for a source
   storage rather than the key-to-key `copy_key` sketched above, because raw
   and datasets are separate `Storage` instances even when they share a bucket.
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
per-run export has no consumer and can go — and it should go in the same deploy
as the `ENTRYPOINT` change the Batch build path needs, since both break the same
deployed contract.

*Both are now done; see "The deploy the plan mispredicted" below.*

**A separate val list in the conversion config.** `split` moves to the build
config, since the split belongs to a dataset version. Relocating it costs
nothing precisely because it has never been usable.

**Aggregate QC in the build.** Per-run QC exists; batch-level properties — the
duplicate seeds, the run whose conversion silently went missing — were caught by
ad-hoc scripts. Whether the build should absorb that is open.

## Validation

Steps 1–3 are implemented and the design above survived mostly intact. Two parts
of it did not, both found by building the thing rather than by reasoning about
it, and both worth recording because the reasoning looked sound.

**The converter could not ask git what commit it was.** The plan assumed the
provenance stamp would mirror the renderer's, which shells out to `git`. The
renderer runs from a checkout; the converter runs from a wheel installed into an
image containing `src`, `configs` and no `.git` at all. A git-based stamp would
have recorded "unknown" for every production conversion and worked only in
development — the exact inverse of the point. The commit is now a build arg baked
into the image, with git as the development fallback, and `scripts/build_image.sh`
computes it so the documented path cannot omit it. Confirmed against a built
image: `.git` absent, stamp resolved from the environment; and with the arg
unset, `commit: null` rather than a plausible-looking wrong answer.

**The first version of that dirty flag was wrong in a way that mattered.**
`git status --porcelain`, run from WSL against a Windows checkout, reports every
file in the repo as modified: the worktree holds CRLF and the index holds LF, so
all 4875 lines differ. Every locally built image would have claimed its labels
came from uncommitted code, discrediting stamps that were in fact exact. Fixed
with `--ignore-cr-at-eol`, scoped to the paths the Dockerfile copies, which is
also the better question — an edited README cannot make the converter disagree
with the commit it names.

Otherwise, as predicted:

- `split_runs` has the multi-run caller it was written for; its existing tests
  pass unchanged, and its rejection of unknown val runs now guards a real
  hold-out list.
- The scene-identity check is the hash rather than the cheap seed comparison.
  The seed form was too strict: the same seed under different randomization is
  genuinely a different scene, and a test pins that it is allowed to straddle
  the split while a true twin is refused.
- Verified against three real corpus runs end to end — real manifests, real
  annotations, real 1.6 MB PNGs: 120 train and 60 val images with matching label
  files, copies byte-identical to their sources, 180 boxes reconciling with the
  manifest totals, no val-run frame in train, and the rebuild refused.
- The full chain now resolves for a real dataset: renderer commit `11075f335e`,
  converter commit `432f7111c5`, builder commit, mask threshold 32.
- 133 unit tests, including local-and-S3 builds producing identical manifests
  apart from `built_at`, and refusals for mixed thresholds, mixed converter
  builds, missing provenance, unconverted and pruned runs, frame-count
  disagreement, duplicate inputs, and a scene spanning the split.

**The corpus was reconverted** to carry provenance, since 69 runs predated the
sidecar and a dataset that cannot attribute its labels defeats the purpose. All
50 now report one converter build and one config, and the reconversion changed
no labels in any run — verified by comparing every annotations file before and
after, which also confirms the deployed code differed from the new image only by
the provenance feature.

**The build role exists** with read on `raw/*` and `datasets/*`, write on
`datasets/*`, and no `ListBucket` at all — a build is told which runs it is
assembling, so it never enumerates. `terraform plan` against the live state
showed 2 to add, 0 to change, 0 to destroy.

One thing the plan asserted without checking: that real builds would run on
Batch. They cannot, as deployed. The image's `ENTRYPOINT` is fixed to
`dronesynth convert --config configs/convert.s3.yaml`, and Batch's
`containerProperties` has no `entryPoint` field to override — only `command`,
which appends. Running a build on Batch therefore needs the entrypoint reduced
to `dronesynth` with the subcommand moved into each job definition, plus a
matching change to the Lambda's command override: a breaking change to the
deployed conversion path, on a deploy where image and job definition must move
together.

That is deferred to ship alongside removing the per-run YOLO export, which is
the other breaking change to the same path — one risky deploy rather than two.
It costs little in the meantime, because `copy_from` removed the reason Batch
looked necessary: with server-side copy the build transfers no image bytes, so
a local build is a few thousand API calls rather than 9 GB of traffic. The role
already trusts `ecs-tasks` so that when the Batch path does land, it needs no
IAM change and gets grants identical to the local one — which was the point of
having a distinct role.

## What the first real build corrected

**v002 exists**: 50 runs, 3000 frames, 3000 boxes, 2400 train and 600 val across
a 40/10 run split, at `s3://drone-synthetic-am/datasets/v002`. Verified
independently of the build's own report — 6002 objects, every image paired with a
label, sampled copies byte-identical to their sources by ETag, sampled labels
reproducing exactly when recomputed from the annotations, and per-run frame and
box counts reconciling against all 50 annotations files. QC flags 35 of 3000
frames (1.2%): 33 tiny boxes at the far end of the approach, which is the small
end we widened the range to get, and 2 frames noting a mask in two pieces, which
instance grouping merges into one box.

**The "no ListBucket" grant broke the build on its first run.** The claim in the
policy comment was true — a build is told its inputs and never enumerates — but
incomplete: without `ListBucket`, S3 answers a HEAD on an absent key with 403
rather than 404, so the role cannot distinguish absent from forbidden. The
pre-flight "does this version already exist" check therefore failed outright
instead of proceeding. The fix keeps the grant and makes the check best-effort,
tolerating `StorageNotPermitted` exactly as ingest already does for its put-only
credentials; immutability rests on the if-absent manifest write, which is what
`write_text_if_absent`'s docstring said all along. Least privilege survived a
real encounter with S3's semantics, and the policy comment now records the price.

**The chain resolves end to end for a real dataset**: renderer `11075f335e`,
converter `432f7111c5`, builder `c7d2a8fb60`, all clean, one converter build and
one mask config across all 50 runs, threshold 32. Scene identities are unique
across the whole corpus, so nothing straddles the split — the duplicate-render
hazard that motivated the check is absent from this build rather than merely
handled.

Worth knowing before the next build: it took 23 minutes. The 6000 requests go
out one at a time, and at S3 latency that is the whole cost — negligible CPU,
no bytes transferred. Batching them would cut it to minutes, which matters once
a corpus is thousands of runs rather than fifty, and is a better first
optimisation than moving the build to Batch.

## Placing frames concurrently

Every key a build writes is derived from a run id and frame index, so the work
is decided before anything is written and no placement depends on or collides
with another. A bounded thread pool is therefore the whole change, and the
ordering that matters is untouched: frames still land before the descriptor and
the manifest, so a version without a manifest is still debris.

Measured on a 180-frame build against real S3, instrumented to record when each
placement starts and ends: peak concurrency 16 of 16 workers, 15.3x effective
parallelism, 3.95 seconds of wall time against 60.7 seconds of summed in-call
time. The mechanism does what it claims.

End-to-end timings on this host are bimodal, and honestly so. Three consecutive
identical builds took 6.1s, 63.1s and 66.8s — 34, 350 and 371 ms per frame
against the 466 ms of the sequential corpus build. The fast mode is the 14x the
instrumentation predicts; the slow mode is close to no gain at all, and its
duration matches the summed in-call time almost exactly, which is what no
overlap looks like. Nothing in the code differs between runs, botocore reports
zero retries and no non-200 responses, and a bare thread pool issuing the same
copies shows the same split, so the cause is below this codebase — most likely
outbound connection setup on a WSL2 host reaching us-east-1 over home
broadband. Left as an observation rather than chased, because even the slow mode
is no worse than sequential and the fix would be somewhere else entirely.

One real defect did surface. botocore pools ten connections by default, and a
caller running sixteen threads against one client does not queue for a slot: it
opens an extra connection and discards it on release, paying a TLS handshake per
request. The pool is now sized above any concurrency here, with a test pinning
that relationship rather than a comment asking future readers to respect it.

## The deploy the plan mispredicted

The plan said a Batch build needs "the entrypoint reduced to `dronesynth` with
the subcommand moved into each job definition, plus a matching change to the
Lambda's command override". The first half is right and the second half does not
work: a `containerOverrides.command` *replaces* the command outright, so
whatever the job definition puts there — the subcommand, the config path —
vanishes the moment a submitter passes a run id. Moving the subcommand into the
job definition and continuing to override the command would have deployed an
image that runs `dronesynth --run-id ... --version ...` and exits on an
unrecognised argument.

Batch parameters are the mechanism that actually fits. The job definition holds
the whole command with `Ref::run_id` placeholders, and a submitter fills those
in through `parameters` rather than replacing anything. That also preserves what
the original design was reaching for — "all a submission contributes is which
run and which dataset version" — instead of forcing every submitter to restate
the subcommand and config path, which is three places to keep in step and the
same mistake in a new shape.

So there are now two job definitions over one image: `dronesynth-convert`
running `convert` under the conversion role, and `dronesynth-build` running
`build` under the build role. The build config is a parameter because there is
one per dataset version and which one this job is assembling is exactly what the
submitter knows.

Dropping the per-run YOLO export shipped in the same change, as intended. It
also removed `split` from the conversion config, which is the honest end of that
thread: the export was the only reason a single-run conversion ever had to
declare a train/val split, and with it gone the knob would have been a lie. A
stale `split:` section in an unedited config is ignored rather than rejected, so
nothing breaks on the way through.

## Clearing the exports the old path left behind

Code stopping writing something does not remove what it already wrote. The 69
`auto-` versions still held their exports: 121 objects each, 8349 in total at
7.6 GB. They are gone now, deleted with an explicit `--include 'auto-*/yolo/*'`
after a dry run confirmed all 8349 lines matched that shape and none mentioned
`annotations`, `raw/` or any `v00*` version. `datasets/` went from 15074 objects
to 6725; the 121 annotation objects and v002's 6001 are untouched.

Three facts made this safe, each checked rather than assumed. The 60 images per
export were byte-identical to the run's own `raw/.../normal/` frames, same ETag
and same length, so `raw/` remained the only copy that ever mattered. The labels
are a pure function of the annotations sitting beside them. And each export's
`dataset.yaml` pointed `val` at an `images/val` holding nothing, which is the
misleading claim this plan set out to delete — a per-run export was never a
dataset, and now nothing implies it was.

What survives per run is the pair a build actually reads: the annotations and
the provenance sidecar. Bucket versioning is on, so these deletes are delete
markers and the bytes are recoverable; reclaiming the space for real needs a
lifecycle rule on noncurrent versions, which does not exist yet.

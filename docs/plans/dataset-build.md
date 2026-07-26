# Plan: assembling a trainable dataset

Status: proposed, 2026-07-26. Written before the work, so the measurements
here describe the problem rather than a result.

Second plan doc; same convention as `instance-boxes.md`. This one exists
because a request that sounded like a config edit — hold out some runs for
validation — turned out not to be expressible in the code as written, and the
reason why is worth recording before choosing a fix.

## What is missing

The first corpus finished on 2026-07-26: 50 runs, 3000 frames, 3000 boxes,
zero empty frames. Every one of those frames converted successfully. None of
them are in a dataset anyone can train on.

Conversion is per-run and always has been. The Lambda submits one job per run
and each job writes its own dataset version, `auto-<run_id>`, holding that
run's frames and nothing else. The bucket currently has 69 of them. They are
useful as staging and as the thing QC reads, but a training set has to span
runs, and nothing assembles one.

The two curated versions that exist, `v00` and `v001`, are earlier test builds.
Each contains a single run — `run_0001`, 150 frames — and in both, every frame
is in `yolo/images/train`. `yolo/images/val` is empty. So the pipeline has
never actually produced a validation split, and the corpus is the first data
large enough to want one.

## Why `split.val_runs` cannot express it

The README says whole runs are held out via `split.val_runs` in the config.
That is the intended design and the config field exists, but setting it today
breaks conversion outright.

`convert_run` asks for an assignment covering exactly one run:

```python
assignments = split_runs([run_id], config.split.val_runs)
```

and `split_runs` rejects any named val run that is not among the runs it was
handed:

```python
unknown = sorted(val_set - set(run_ids))
if unknown:
    raise SplitError(f"val_runs not among the input runs: {unknown}")
```

With ten runs named and one run being converted, `unknown` is the other nine
and the job fails. Naming a single run does not help either: converting any of
the other 49 still leaves that one unknown. **Any non-empty `val_runs` fails
every conversion**, which is every run the auto path touches.

The check is not wrong. `test_unknown_val_run_rejected` pins it deliberately,
and it catches a typo in a hand-written val list — a real hazard, since a
misspelled run id would otherwise silently produce an empty validation set.
The problem is that `split_runs` was written for a caller that assembles many
runs at once, and that caller was never built. Its only real caller passes one
run.

## The decision

**A split is a property of an assembled dataset, not of a conversion.**

Converting one run answers "what is in these frames". Splitting answers "which
runs does this dataset train on and which does it validate against", and that
question does not exist until something decides which runs the dataset
contains. Per-run conversion has no split to make, and the current code only
appears to give it one because a lone run trivially lands in train.

So the missing piece is an assembly step — a `build` command that takes a set
of runs and a version name, applies the split across them, and writes one
dataset. `split_runs` then gets the caller it was designed for, strict
validation and all, and the typo check starts protecting something.

### Build from annotations, not by re-converting

The canonical per-frame JSON is already written and already in the bucket, one
file per run under `auto-<run_id>/annotations/`. A build does not need to
re-threshold a single mask; it needs to read those annotations, copy the normal
frames into `images/{train,val}/`, write label files, and emit `dataset.yaml`.

This is what "canonical JSON annotations; YOLO is an export" already claims in
the README, and building this way is the first thing that would make the claim
load-bearing. It also means a re-split is cheap — changing which runs are held
out re-copies files but recomputes no labels — and that a build cannot disagree
with the QC report the annotations came from.

The copies can be server-side within the bucket, so 3000 frames never round-trip
through the machine running the build.

## Why not the alternatives

**Relax `split_runs` to treat `val_runs` as a policy list.** Ignore names that
are not in the input, so a per-run conversion just asks "is this run in val".
It is a two-line change and it makes the config field work as the README
describes. Rejected as the primary fix because it produces single-run datasets
that are entirely val and entirely empty of train, which is a strange artifact,
and because it silently discards the typo check at exactly the moment the val
list gets long enough to typo. It also does not produce a trainable dataset,
which is the actual goal — it only relabels which folder a lone run lands in.
Worth doing as a follow-on once `build` exists, if per-run versions should
reflect their eventual side.

**Have `convert_run` pass the intersection** — `val_runs ∩ {run_id}` — so the
strict check always passes. Same effect as the above with less honesty: the
check remains in the code, still looks like it validates something, and no
longer can fail.

**Set `val_runs` and re-convert only the held-out runs.** Does not work at all,
per the mechanism above, and would not produce a merged dataset even if it did.

**Skip the split; train on everything and evaluate on real footage.** Defensible
in principle, since the real branch is where the distribution that matters
lives. Rejected because the real branch has no labels yet — that is recorded in
the README as the reason it is not trainable — so there would be nothing to
evaluate against, and a synthetic val set is the only measurement available
until labeling exists.

## Which runs to hold out

Recession and crossing clips come from different configs and are genuinely
different distributions, so the hold-out should be proportional to both rather
than a flat sample of 50. At 20%, that is 7 of 35 recession and 3 of 15
crossing.

Proposed: every fifth seed in each range.

| | seeds |
| --- | --- |
| recession val | 1102, 1107, 1112, 1117, 1122, 1127, 1132 |
| crossing val | 2102, 2107, 2112 |

600 val frames against 2400 train. Seeds are independent draws, so a contiguous
block would be just as valid statistically; spreading them is only a guard
against ordering effects nobody has checked for, such as the two overlapping
render batches on the day having differed in some way not visible in the QC.

Worth stating plainly: 10 runs of one drone model on one map is a thin
validation set. It measures whether the converter and the renderer agree, not
whether a detector generalizes. That is the right thing to measure right now,
and it should not be mistaken for the other one.

## Open questions

- Should `build` re-run QC across the assembled set, or is per-run QC enough?
  Aggregate QC is what caught the duplicate seeds and the run whose conversion
  had silently gone missing, and both were properties of the batch rather than
  of any run.
- Should the val list live in the conversion config, or in the build invocation?
  Config makes it reproducible and reviewable; a flag makes it obvious that it
  belongs to one dataset version rather than to conversion generally. Config
  seems right, given that a dataset version is meant to be fully determined by
  (input runs, conversion config).
- Do the 69 `auto-<run_id>` versions stay once curated builds exist? They cost
  storage and duplicate every frame. Keeping them is defensible while they are
  the only per-run QC artifact.
- What identifies the input set of a build? Listing 50 run ids in a config is
  unreadable. A seed range or a `generator` commit filter would be terser, and
  the manifest now carries enough to support either.

## Validation

Not yet done. When implemented:

- `split_runs` gets a real multi-run caller, and the existing tests keep
  passing unchanged.
- A build over the 50-run corpus produces 2400 train and 600 val images with
  matching label counts, and no run id appears on both sides.
- Frame counts reconcile against the QC reports the annotations came from:
  3000 images, 3000 labels, 3000 boxes.
- Re-running the same build produces byte-identical output, per the
  determinism the README claims for a dataset version.

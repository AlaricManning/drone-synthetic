# Plan: training a detector on the corpus

Status: planning, 2026-07-28. Nothing implemented. Written before the work, like
`dataset-build.md`, so the body is a design record rather than a measurement and
the numbers in it are inputs rather than results.

One decision is settled: the training code goes in a separate repository under
AGPL-3.0, not in this one — see [the licence
section](#the-licence-decides-where-the-code-lives). Resolution and model choice
are open, and the resolution numbers below are the reason that is not a detail.

Third plan doc. This one exists because "all that is left is training" is very
nearly true and the ways it is not true are the interesting part. v004 is
already in the shape a trainer wants — `datasets/v004/yolo/` holds 7200 training
and 1800 validation images with matching labels, one class, a `dataset.yaml`
Ultralytics can read unmodified — and there is a 16 GB RTX 5080 to run it on. The
mechanical work is an afternoon.

The parts that are not mechanical are three: the licence decides where the code
is allowed to live, the first model is an instrument rather than a product, and a
number nobody can reproduce is not a result. This project has spent considerable
effort making every frame traceable to the commit that rendered it. Training is
the first step that will produce a figure somebody quotes, and it would be a poor
place to drop that discipline.

## What the first model is for

Not deployment. Three questions it can answer that nothing else can:

**Is the data learnable at all?** Every stage so far has been checked for
internal consistency — masks align with frames, boxes derive from masks, splits
contain the runs they claim. None of that establishes that a detector can fit it.
A model that trains to a sensible loss curve is the first end-to-end evidence
that the mask threshold, the box extraction and the label format are mutually
coherent rather than merely individually plausible.

**Where does accuracy fall off?** This is the real product of the first run, and
it is why the corpus records what it records. Accuracy sliced by distance, by
weather preset and by measured contrast is a description of what the dataset
teaches and what it does not.

**What should the contrast threshold be?** `ct-build-filter` has been parked
since the contrast metric landed, because a threshold chosen without a training
result is a guess dressed as a decision. The corpus already shows the naive
answer is wrong: 11.0% of `Clear_Skies` frames flag as low contrast against 0.0%
of `Overcast`, `Rain`, `Snow` and `Sand_Dust_Light`, because the metric measures
which side of the background the target sits on rather than how far from it. A
flat cut would delete good clear-sky frames and keep obscured dusty ones. What
would settle it is knowing which flagged frames a detector actually gets wrong.

What the first model cannot answer is how it performs on real video, and the gap
is not small. See "What will make the numbers look better than they are".

## The licence decides where the code lives

This is the constraint most likely to be discovered too late, because it does not
announce itself until the code is written and sitting in the wrong repository.

Ultralytics YOLO is AGPL-3.0. `object-tracker` already carries AGPL-3.0 for
exactly that reason and says so in as many words — inherited rather than chosen.
Its licence note also asserts something about the siblings:

> This applies to this repository alone. The sibling repositories, drone-synthetic
> and object-tracker-pipeline, depend on nothing under AGPL and grant no licence
> — all rights reserved there.

That sentence is currently true and a training script in this repository would
make it false. `drone-synthetic` is publicly readable with all rights reserved.
Publishing source that combines with AGPL code is distribution of a combined
work, so importing `ultralytics` here would either oblige this repository to be
AGPL-3.0 or put it in violation. Private use would not trigger it; a public
repository is not private use.

Three ways out, and they are genuinely different choices rather than shades of
one:

**Train inside `object-tracker`.** Already AGPL, already depends on Ultralytics,
so nothing changes licence-wise and the model ends up next to the code that runs
it. Against it: that repository is a real-time webcam tracker with KLV metadata
handling and an S3 uploader. A training harness shares no code with it and would
sit there as an unrelated concern in a repository whose README is about tracking.

**A new repository under AGPL-3.0.** One concern per repository, an honest licence
from the first commit, and `drone-synthetic` stays clean. It is the option that
keeps the dependency boundary visible: this repository produces datasets and
grants no licence, that one consumes them and is AGPL because of what it imports.
The cost is another repository to maintain.

**Do not use Ultralytics.** A detector under Apache-2.0 or BSD — torchvision's
detection models, RT-DETR, YOLOX — leaves every repository unencumbered and
removes any question about commercial use later. The cost is giving up the most
convenient trainer in the ecosystem and the one already proven in
`object-tracker`, and writing more glue.

### Decided: a separate repository under AGPL-3.0

Chosen 2026-07-28, before any code was written, because moving a repository's
licence afterwards is much harder than choosing it.

What follows from it. The new repository is AGPL-3.0 from its first commit rather
than relicensed later, and it depends on `drone-synthetic` only through published
dataset artefacts in S3 — it reads `datasets/v004/yolo/` and never imports this
package, so no AGPL obligation propagates backwards. `drone-synthetic` keeps all
rights reserved and `object-tracker`'s licence note stays true as written, which
is the specific property this decision was made to preserve.

It also accepts a consequence: staying with Ultralytics means commercial use of
anything trained there needs a licence from them, including the weights on the
reading described below. That is a live cost rather than a hypothetical, and the
escape hatch is the third option above — a detector under Apache-2.0 or BSD —
which remains available precisely because the training code will be new and small
rather than entangled with the rest of the project.

One wrinkle to note rather than resolve: Ultralytics takes the position that
weights trained with their code are a derivative work. That reading is contested
and this document does not need to settle it, but it means a fine-tuned
checkpoint should be treated as carrying the same terms as the trainer that
produced it. Deploying it inside `object-tracker` is consistent, since that is
AGPL already. Selling it is not, without a licence.

## What v004 already provides, and what it does not

Provided, and verified: `yolo/dataset.yaml` with `names: {0: drone}`, 7200 train
and 1800 val images with a label file each, boxes normalised and valid, a split
drawn at run level so no trajectory appears on both sides, and a dataset manifest
naming every run with the weather preset it was rendered under. Total 16.4 GB
across 18,002 objects.

Not provided, and each is a decision the harness has to make: an augmentation
policy, a training resolution, and any notion of a negative example. There is no
frame in the corpus without a drone in it.

## Resolution is the one hyperparameter that could invalidate the run

Ultralytics defaults to 640. These frames are 1920 wide, so the default is a 3x
reduction applied to a corpus deliberately built to include targets at the edge
of visibility. Measured over all 150 runs from the QC reports, at native
resolution:

| | box area | side |
| --- | --- | --- |
| smallest box in the corpus | 40 px² | 6.3 px |
| median run's smallest box | 134 px² | 11.6 px |
| largest box in the corpus | 313,532 px² | 560 px |

The finest stride in a YOLO detection head is 8, so a target much under 8 px has
no feature-map cell to be found in. Applying each candidate resolution to those
two small figures:

| imgsz | corpus smallest | median run's smallest |
| --- | --- | --- |
| 640 | 2.1 px | 3.9 px |
| 960 | 3.2 px | 5.8 px |
| 1280 | 4.2 px | 7.7 px |
| 1920 | 6.3 px | 11.6 px |

Two conclusions, and the second is the uncomfortable one.

The default is clearly unusable. At 640 the smallest target in a typical run is
about 4 px, so the far end of nearly every run falls below what the head can
represent. Training there would discard precisely the long-range frames the
scale-range work existed to produce, and it would present as "the model cannot
see distant drones" — which reads like a data problem and is not one.

But the tail is marginal even at native resolution. 6.3 px at 1920 is already at
the stride limit, and the median run's smallest box only reaches 11.6 px. So this
is not merely a resize setting to get right: the corpus's far endpoint produces
targets that a standard stride-8 head cannot learn from at any resolution the
frames actually have. That is worth knowing before it is discovered as a mystery
in a per-distance accuracy table.

Which leaves a real choice rather than a default to correct. Training at 1280 or
1920 keeps most of the range detectable and costs memory and time roughly with
the square of the side. A head with a stride-4 level, which Ultralytics exposes
as the P2 variants, is the change that actually addresses small targets rather
than working around them. Tiling preserves native resolution at the cost of
pipeline complexity and label bookkeeping. And pulling the far endpoint in is a
legitimate answer too — `drone-synth-render`'s `scale-cap.md` chose 400 m on
visibility grounds, not on whether a detector head can resolve the result, and
these numbers are new evidence for that decision rather than against it.

What should not happen is training at 640, observing that distant drones are
missed, and concluding anything about the data.

## Making a result reproducible

The chain already runs: a render records the generator commit, a conversion
writes a provenance sidecar with the converter commit and mask configuration, and
a build snapshots both plus the run list and split. Training is where it would
break, because a training run has more inputs than any earlier stage and none of
them are currently written down anywhere.

A training run should emit a manifest of its own, on the same principle as the
others — enough to reproduce the number without consulting anybody's memory:

- the dataset version and a digest of its manifest, so the exact 7200/1800 split
  is identified rather than described
- the training code commit, and whether the tree was dirty
- the base checkpoint, by name and hash
- every hyperparameter that was not a default, and the resolution in particular
- the random seed
- a hash of the resulting weights
- the metrics, including the sliced ones below

Written last, like the run manifests, so a manifest's presence means the run
finished. The alternative is a directory of checkpoints whose provenance is a
filename, which is what this project has spent its whole history avoiding.

## The metrics that matter are the sliced ones

Aggregate mAP over this validation set will be dominated by easy frames, because
a recession spends most of its length nearer than its far endpoint. A single
number would mostly describe the near half of the trajectories.

The corpus records exactly what is needed to do better. The dataset manifest
names each run's preset, so results group by weather condition. Each box carries
a measured contrast, so results group by how visible the target was. Distance is
recoverable per run from the trajectory endpoints, and the recession-versus-
crossing distinction is in the build config. The interesting output of a training
run is therefore a table rather than a scalar: accuracy by distance bucket, by
preset, by contrast bucket, and by clip kind.

One caution about how hard those numbers can be pushed. The validation set is 30
runs, three per preset, 60 frames each. Frames within a run are consecutive views
of one trajectory and are nowhere near independent, so a per-preset figure rests
on three trajectories, not 180 samples. Those numbers will be indicative and
should be reported as such; distinguishing `Rain` from `Rain_Light` on that
evidence is not something this split can support.

## What will make the numbers look better than they are

Worth writing down in advance, because the temptation to read a high mAP as
success will be strongest at the moment the first number appears.

**There is nothing to reject.** Every frame is a drone against sky. The model
never sees a dark high-frequency speck that is not a drone, so it has no
opportunity to produce the false positives that make precision meaningful. See
`drone-synth-render`'s `background-diversity.md`, which records this as a known
and deliberate limitation. Precision will look excellent for a reason that has
nothing to do with the model being good.

**There is one drone.** Every frame is `white_drone`. A model can fit that
airframe's appearance rather than learning what drones look like, and no held-out
run will catch it because the hold-out has the same aircraft in it.

**9000 frames is 150 samples.** Sixty frames from one trajectory are not sixty
independent observations. The effective size of this dataset is closer to its run
count than its frame count, for both training and evaluation.

**Three of the ten conditions are two conditions.** Snow does not reach the frame
under the cloud its own preset supplies, so the `Snow` and `Snow_Light` runs are
in practice more heavy-overcast runs. The manifest's `preset` field records what
was requested, which is not the same as what is visible.

The honest expectation is a high mAP that mostly measures how much easier this
task is than the real one. The useful signal is the shape of the falloff, not its
height.

## What lands, in order

Now that placement is settled this can be sequenced. The ordering principle is
that the two decisions capable of invalidating a run — where the code lives and
what resolution it trains at — come before anything expensive.

1. Create the new repository, AGPL-3.0 in the first commit, with a README that
   says why it is AGPL and that its siblings are not.
2. A dataset pull that fetches `datasets/v004/yolo/` to a local cache and
   verifies it against the dataset manifest, so a training run cannot silently
   train on a partial download. Cache, not re-fetch: 16.4 GB per experiment is a
   waste and an invitation to skip the verification.
3. Settle resolution against the numbers in this document, on a short run rather
   than by argument. A few epochs at 640 and at 1280 will show the far-distance
   difference plainly, and it is cheaper to measure than to debate.
4. The training manifest, written before the first run anybody intends to quote.
   Retrofitting provenance means the first interesting result is the one that
   cannot be reproduced.
5. The first real run, and the sliced metrics table rather than a scalar.
6. Feed the contrast slice back into `ct-build-filter` here, either as a chosen
   threshold or as a finding that per-box contrast is the wrong quantity to cut
   on.

## What would count as done

- A training run that can be repeated from its manifest alone and lands within
  noise of the original.
- A resolution chosen against measured target sizes rather than inherited from a
  default.
- Accuracy sliced by distance, preset and contrast, with the run-count caveat
  stated wherever the slices are reported.
- Enough of that to choose the contrast threshold in `ct-build-filter`, or to
  establish that per-box contrast is the wrong quantity to cut on and say what
  would be better.
- ~~Training code in a repository whose licence is correct for what it
  imports.~~ Decided: a separate AGPL-3.0 repository. Done when it exists and
  this repository still grants no licence.

## Open questions

- Which detector and which size. `yolo11n` matches what `object-tracker` already
  runs, which is an argument for it beyond convenience; a larger backbone may be
  needed at 1280 or above.
- Fine-tune from COCO or train from scratch. COCO has no drone class and the
  nearest analogues are `airplane` and `bird`, so the transferable part is
  low-level features rather than anything class-specific.
- How much colour and geometric augmentation. Heavy photometric augmentation
  would partly undo the atmospheric randomisation the corpus was built to
  provide, which is an odd thing to pay render time for and then destroy.
- Whether to add frames with no drone in them, and whether they can come from
  this generator at all given that every map it has is sky.
- Where training runs once it outgrows one machine. The Batch infrastructure
  exists but has no GPU queue, and a 16 GB card is enough for a long time at this
  dataset size.
- Whether a small hand-labelled set of real video should exist as a test set
  before any of this, on the grounds that a synthetic-only evaluation cannot
  measure the thing anybody cares about.

## Deferred, deliberately

Background diversity and a second airframe are both larger than this plan and
both would change what a trained model means. Neither is a prerequisite for the
first training run, because the first run's purpose is to measure the pipeline
and the dataset rather than to produce something deployable. They are
prerequisites for believing a good number.

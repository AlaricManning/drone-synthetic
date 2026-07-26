# Plan: label objects, not islands

Status: implemented, 2026-07-26. Frozen — this records what was decided and
measured on that date, not the code as it stands later.

First plan doc in this repo; the convention is borrowed from `drone-synth-render`,
where `docs/plans/` holds the reasoning behind changes big enough that the
reasoning outlives the diff. Written *after* the work rather than before it,
because this began as a bug report rather than a piece of planned work.

## What went wrong

`drone-synth-render` #17 widened the drone's heading from nose-on ±25° to the
full circle. The first batch afterwards, seeds 500–502, converted cleanly on two
runs and badly on the third:

```
run run_20260726_013904_502: 60 frames, 73 boxes, 0 empty frames
  box area 16..208640 px
  26 flag(s) — frame 20: 2 boxes in frame / frame 20: tiny box 13x4 ...
```

Seventy-three boxes for sixty frames of a scene holding one drone. Thirteen
frames each carried an extra label of 3–13 px, and every one of them sat
entirely inside the correct box. The YOLO labels asserted that a 13×4 sliver in
the middle of a drone was itself a whole drone — the precise thing that teaches
a detector to fire on specks, and that gives a tracker phantom targets to hold.

The cause is that at oblique headings a propeller blade's supporting arm
projects thinner than a pixel and drops out of the render, stranding the blade
as its own island. `extract_boxes` emitted one box per island.

## The decision

**Group mask pixels by object, not by connectivity.**

The mask pass paints only the drone actor and hides everything else, so which
object a pixel belongs to is not something the converter has to infer —
connectivity is a fact about rasterisation, and using it as the grouping key was
always a proxy that happened to hold.

Components stay as a primitive in `extract_boxes`; `extract_instances` groups
them into the unit a detector is actually taught. Today the grouping is trivial,
because one drone means every island is part of it.

### Why not the alternatives

Four options were on the table. The measurements that separated them: all 13
stray boxes were fully contained in the main box, and the gaps splitting the
blade from the airframe were 1–2 px.

**Morphological closing** was the leading candidate and is the one this replaces.
Dilate-then-erode with a 5×5 kernel would bridge those gaps. It fixes the cause
at the pixel level and is asset-agnostic. But the kernel is in absolute pixels
while the gap scales with apparent size — 1–2 px on a 42 px drone at 110 m is
proportionally larger on the same airframe at 11 m, and different again on a
fixed-wing's tail boom. It would need re-tuning per asset and per distance band.
Worse, it cannot survive the background work: a drone crossing behind a branch
splits by tens of pixels, and no kernel bridges that without also merging things
that should stay apart.

**Dropping boxes contained in a larger box** clears 13 of 13 here and is three
lines. It was rejected because the containment is luck of this geometry: when
the *outermost* blade detaches, the fragment lands outside the main box and the
main box is short of the airframe. The filter does nothing in exactly the case
where the label is most wrong. `test_a_piece_outside_the_body_still_widens_the_box`
pins that case down.

**Raising `min_box_area`** filters on the wrong property — these boxes are wrong
because they are duplicates, not because they are small — and it has an expiry
date, since widening the distance range shrinks real drones into the filtered
band.

**Motion blur in the renderer** addresses the true root cause and remains the
right long-term answer for thin moving structure, but it is deferred in
`drone-synth-render`, does nothing for frames already rendered, and would break
the `1/distance` assumption in `check_approach.py`.

### It is also the multi-target design

The objection to merging is that it hard-codes one drone per frame. That is true
of "union everything" but not of "group by instance": when per-drone mask values
arrive, the key becomes the value and the shape of the code is unchanged.
`extract_instances` returns a list for that reason.

The mechanism to reach for then is UE's Custom Depth/Stencil, which writes
integer IDs with no antialiasing between them. Distinct colours would not work —
antialiased edge pixels blend two instances' colours and become unassignable.

## What the fix uncovered

Re-converting 502 gave the right box count immediately, and a fill ratio that
had collapsed from 0.37 to 0.08 on the first few frames. Grouping was stretching
boxes across the frame.

The mask frames turned out to contain faint static specks reading 13–31,
hundreds of pixels from the drone, on pixels whose RGB is tan terrain. They
decay from 21 to 1 as the drone recedes and are absent from seeds whose drone
never gets close. **The mask material bounces light onto the ground**, so the
mask pass is not the clean matte the pipeline assumed.

This had always been there. It was invisible only because each speck was too
small to clear `min_box_area` as its own component — the old bug was hiding it.
Grouping made it load-bearing.

Threshold 32 rejects it. Measured on 502:

| threshold | boxes | worst frame's islands | min fill | box width f0/f20/f40 |
| --- | --- | --- | --- | --- |
| 12 | 60 | 28 | 0.076 | 652/112/63 |
| **32** | **60** | **3** | **0.357** | **650/112/63** |
| 64 | 60 | 3 | 0.343 | 650/112/61 |
| 96 | 60 | 4 | 0.333 | 648/110/61 |

Two pixels of box width buys the bounce being gone. Above 32 it starts chipping
the airframe apart instead.

Worth recording that this reverses a conclusion reached a day earlier. Box width
is genuinely identical from `>2` to `>32`, and that measurement was used to argue
the threshold barely mattered and to align the render repo's QC down to 12. The
measurement was right and the conclusion was wrong: width only ever tracked the
largest island, so it could not see the thing the threshold was actually holding
back.

## Follow-up for drone-synth-render

The bounce should not be in the mask render at all. The mask pass wants unlit
shading or Lumen disabled, so the only lit pixels are the drone's own. Until
then the threshold is doing that job, which works but couples the converter to a
renderer artefact.

Also unchanged: the propellers still fragment. Grouping makes it harmless to the
box, and the `mask in N pieces` flag reports it rather than hiding it — 20 such
flags remain on 502 at threshold 32. Motion blur is the fix.

## Validation

- 76 tests pass. New coverage for a stranded piece joining its object, a piece
  *outside* the body widening the box, an occluded drone staying one box, fill
  ratio counting pixels rather than the span, and `min_box_area` judging the
  assembled object rather than its pieces.
- All three runs of the 500–502 batch re-converted at threshold 32: 60 boxes for
  60 frames, zero frames with more than one box, zero low-fill flags. Seed 500
  is flag-free entirely, 501 has one fragmentation flag, 502 has twenty.

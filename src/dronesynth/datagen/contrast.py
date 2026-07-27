"""How visible a labelled object actually is in the render it labels.

Every other number in an annotation comes from the mask, which is rendered with
the drone isolated in clear air. That is what makes the silhouette clean, and it
is also why the mask cannot say whether the object is *visible*: it holds a
crisp 55-pixel drone whether the normal render shows one or shows haze.

The renderer measured the consequence. At 400 m, sweeping Ultra Dynamic
Weather's fog, the drone's contrast against the sky went +15.7, +8.0, −4.1 —
crossing zero around fog 0.6 — while the mask held exactly 55 pixels at every
step. A label with no contrast behind it is not a hard example, it is a wrong
one, and nothing in the mask distinguishes the two. See the visibility probe in
drone-synth-render's docs/plans/domain-randomization.md.

So this is measured here rather than in the renderer, and acted on later still.
The raw pairs are kept in S3 permanently, conversion is deterministic and
re-runnable, and both the metric below and any threshold over it are judgements
that will be revised — the metric already changed once during the probe, from
an unsigned whole-mask difference to the signed core-pixel one here, and that
changed the conclusion. Measuring at conversion means refining it costs a
re-convert rather than a re-render, and backfills every run already in the
bucket. Thresholding at build time means the cut can move for free.

Why the measurement is shaped this way
--------------------------------------
*Signed.* A drone brighter than the sky and a drone darker than it are equally
detectable, but fog moves an object across that boundary, and an absolute
difference reports the crossing as a bounce off zero rather than a passage
through it. The sign is what says which side of its background the object is on.

*Core pixels, not the whole mask.* At 400 m the drone covers 55 pixels of which
roughly 10 are fully opaque; the rest are anti-aliased edge whose colour is
mostly sky. Averaging all of them drags any measurement toward zero contrast
regardless of what the air is doing, so the object's own pixels are the ones
above ``CORE_ALPHA``. Far enough away nothing clears that bar, and the most
covered pixels are used instead so the number stays defined.

*A ring, not the frame.* What a detector resolves is the edge between object
and immediate background, and the sky two hundred pixels away is not what the
object is competing with. The ring excludes *every* object's pixels rather than
only this one's, so a second drone parked alongside cannot be mistaken for sky.
"""

from __future__ import annotations

import cv2
import numpy as np

from dronesynth.datagen.boxes import DetectedBox

# Mask value at or above which a pixel is the object rather than a blend of it
# with the background. 200 of 255 leaves room for the renderer's own filtering
# without admitting half-covered edge pixels.
CORE_ALPHA = 200

# Width of the background annulus, in pixels. Wide enough to hold a usable
# sample of sky beside a 13-pixel drone, narrow enough to stay local.
RING_PX = 15

# Below this many core pixels, fall back to the most covered ones. Eight is
# enough to average without being one hot pixel, and small enough that a drone
# at the far end of the range still gets a number.
MIN_CORE_PX = 8


def metric_config() -> dict[str, int]:
    """The metric's parameters, for the provenance record.

    Contrast is only comparable across runs measured the same way, so these ride
    along with the mask settings and the build refuses to mix runs that disagree.
    """
    return {"core_alpha": CORE_ALPHA, "ring_px": RING_PX, "min_core_px": MIN_CORE_PX}


def to_grey(image: np.ndarray) -> np.ndarray:
    """Channel mean, deliberately not a luma weighting.

    Luma weights green because human vision does, and nothing downstream of
    here is human vision. A plain mean treats the three channels a detector
    receives alike, and has the practical virtue of being obvious.
    """
    if image.ndim == 2:
        return image.astype(np.float32)
    return image[:, :, :3].astype(np.float32).mean(axis=2)


def mask_alpha(mask: np.ndarray) -> np.ndarray:
    """Per-pixel coverage, on the same convention :func:`binarize_mask` uses.

    That function calls a pixel drone when *any* channel clears the threshold,
    so the strongest channel is the coverage this has to agree with.
    """
    if mask.ndim == 2:
        return mask
    return mask[:, :, :3].max(axis=2)


def _ring(objecthood: np.ndarray, every_object: np.ndarray) -> np.ndarray:
    grown = cv2.dilate(
        objecthood.astype(np.uint8), np.ones((RING_PX, RING_PX), np.uint8)
    ).astype(bool)
    return grown & ~every_object


def _core(objecthood: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    core = objecthood & (alpha >= CORE_ALPHA)
    if core.sum() >= MIN_CORE_PX:
        return core
    covered = alpha[objecthood]
    if covered.size == 0:
        return core
    keep = min(MIN_CORE_PX, covered.size)
    floor = np.partition(covered, -keep)[-keep]
    return objecthood & (alpha >= floor)


def measure_contrast(
    grey: np.ndarray,
    alpha: np.ndarray,
    every_object: np.ndarray,
    box: DetectedBox,
) -> float | None:
    """Signed contrast between one object's pixels and the background beside it.

    ``every_object`` is the whole frame's drone map, used to keep other objects
    out of the ring; the box selects which of them this measurement is about.
    Returns ``None`` when the object or its surroundings cannot be sampled --
    an object filling the frame has no background to compare against, and that
    is a missing measurement rather than a contrast of zero.
    """
    region = np.zeros_like(every_object)
    region[box.y : box.y + box.h, box.x : box.x + box.w] = True
    objecthood = every_object & region
    if not objecthood.any():
        return None

    ring = _ring(objecthood, every_object)
    if not ring.any():
        return None

    core = _core(objecthood, alpha)
    if not core.any():
        return None

    return float(grey[core].mean() - grey[ring].mean())

import numpy as np

from dronesynth.datagen.boxes import DetectedBox, binarize_mask, extract_instances
from dronesynth.datagen.contrast import (
    CORE_ALPHA,
    MIN_CORE_PX,
    mask_alpha,
    measure_contrast,
    metric_config,
    to_grey,
)


def scene(object_grey, background_grey, alpha=255, size=12, h=96, w=96):
    """One square object of a given brightness on a flat background."""
    normal = np.full((h, w, 3), background_grey, dtype=np.uint8)
    normal[40 : 40 + size, 40 : 40 + size] = object_grey
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[40 : 40 + size, 40 : 40 + size] = alpha
    return normal, mask


def measure(normal, mask, threshold=32):
    binary = binarize_mask(mask, threshold)
    boxes = extract_instances(binary, min_box_area=1)
    assert boxes, "fixture produced no object"
    return measure_contrast(to_grey(normal), mask_alpha(mask), binary, boxes[0])


def test_an_object_darker_than_its_background_is_negative():
    assert measure(*scene(object_grey=100, background_grey=150)) == -50.0


def test_an_object_brighter_than_its_background_is_positive():
    """Sign is the point: fog carries an object across zero, not into it."""
    assert measure(*scene(object_grey=200, background_grey=150)) == 50.0


def test_an_object_matching_its_background_has_no_contrast():
    assert measure(*scene(object_grey=150, background_grey=150)) == 0.0


def test_anti_aliased_edge_does_not_dilute_the_measurement():
    """The failure this metric exists to avoid.

    A distant drone is mostly edge pixels whose colour is largely background.
    Averaging the whole mask would report almost no contrast for an object that
    is plainly there, so only fully covered pixels are measured.
    """
    normal, mask = scene(object_grey=100, background_grey=150, size=12)
    # Ring the object in half-covered pixels that read as background colour.
    mask[39:53, 39:53] = np.where(mask[39:53, 39:53] > 0, mask[39:53, 39:53], 128)
    normal[39:53, 39:53] = np.where(normal[39:53, 39:53] == 100, 100, 145)

    assert measure(normal, mask) == -50.0


def test_a_distant_object_with_no_fully_covered_pixels_still_measures():
    """Far enough out nothing clears CORE_ALPHA, and None would lose the frame."""
    normal, mask = scene(object_grey=100, background_grey=150, alpha=CORE_ALPHA - 20)

    value = measure(normal, mask)

    assert value is not None
    assert value == -50.0


def test_the_fallback_prefers_the_most_covered_pixels():
    normal, mask = scene(object_grey=150, background_grey=150, size=8)
    # A handful of well-covered pixels over a dark object, the rest barely
    # covered over background-coloured pixels. The dark ones are the drone.
    mask[40:48, 40:48] = 60
    mask[40:42, 40:44] = 190  # 8 px, still under CORE_ALPHA
    normal[40:42, 40:44] = 90

    value = measure(normal, mask)

    assert value == -60.0


def test_another_object_in_the_ring_is_not_mistaken_for_background():
    """Two drones side by side: neither may serve as the other's sky.

    Grouping merges everything in frame into one box today, so the per-object
    box is built directly -- this guards the ring for the multi-instance case
    the metric is written to survive rather than for one that exists yet.
    """
    normal, mask = scene(object_grey=100, background_grey=150, size=10)
    normal[40:50, 54:64] = 20  # a second, much darker object, 4 px away
    mask[40:50, 54:64] = 255

    binary = binarize_mask(mask, 32)
    first = DetectedBox(x=40, y=40, w=10, h=10, mask_area=100)
    value = measure_contrast(to_grey(normal), mask_alpha(mask), binary, first)

    # Background is 150 and the object is 100. Had the neighbour's 20 leaked
    # into the ring it would have pulled the reference down and shrunk this.
    assert value == -50.0


def test_an_object_with_no_background_beside_it_is_unmeasured():
    """Not zero contrast -- no measurement. The distinction is the point."""
    normal = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.full((32, 32, 3), 255, dtype=np.uint8)

    assert measure(normal, mask) is None


def test_metric_config_names_every_parameter_that_moves_the_number():
    """It rides in the provenance, so a forgotten key is a silent mismatch."""
    assert metric_config() == {
        "core_alpha": CORE_ALPHA,
        "ring_px": 15,
        "min_core_px": MIN_CORE_PX,
    }


def test_grey_is_a_plain_channel_mean():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = (30, 60, 90)

    assert to_grey(image)[0, 0] == 60.0


def test_alpha_follows_the_binarize_convention_of_any_channel():
    mask = np.zeros((1, 1, 3), dtype=np.uint8)
    mask[0, 0] = (0, 0, 220)

    assert mask_alpha(mask)[0, 0] == 220

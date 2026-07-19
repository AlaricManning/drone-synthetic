"""Mask render -> bounding boxes.

The mask render shows the drone against black. A pixel belongs to the drone
if any channel exceeds the configured threshold; connected components of
those pixels become boxes. Components whose bounding box is smaller than
``min_box_area`` pixels are dropped as mask noise (isolated bright pixels
from anti-aliasing at the silhouette edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectedBox:
    """A connected component's bounding box, in pixel coordinates.

    ``mask_area`` is the component's pixel count; ``fill_ratio`` is
    mask_area / box area — a drone silhouette fills a fair fraction of its
    box, so a very low ratio flags a suspicious detection.
    """

    x: int
    y: int
    w: int
    h: int
    mask_area: int

    @property
    def box_area(self) -> int:
        return self.w * self.h

    @property
    def fill_ratio(self) -> float:
        return self.mask_area / self.box_area


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image, read from metadata without decoding pixels."""
    with Image.open(path) as image:
        return image.size


def load_mask(path: Path) -> np.ndarray:
    # IMREAD_COLOR drops the alpha channel EasySynth writes (opaque everywhere,
    # so it would put every pixel above any threshold)
    mask = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"cannot read mask image: {path}")
    return mask


def binarize_mask(mask: np.ndarray, threshold: int) -> np.ndarray:
    """Boolean drone/background map: any color channel above threshold is drone."""
    if mask.ndim == 3:
        return (mask[:, :, :3] > threshold).any(axis=2)
    return mask > threshold


def extract_boxes(binary: np.ndarray, min_box_area: int) -> list[DetectedBox]:
    """Connected components of a boolean mask, as boxes, largest first."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    boxes = []
    for label in range(1, count):  # label 0 is background
        x, y, w, h, area = stats[label]
        box = DetectedBox(x=int(x), y=int(y), w=int(w), h=int(h), mask_area=int(area))
        if box.box_area >= min_box_area:
            boxes.append(box)
    return sorted(boxes, key=lambda b: b.mask_area, reverse=True)

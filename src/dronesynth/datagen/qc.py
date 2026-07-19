"""Quality control over a converted run.

Nothing downstream in this pipeline trains on the data, so this report and
the debug renders are the evidence the labels are good. Stats summarize the
run; flags mark individual frames worth a human look — the point is to eyeball
flagged frames, not all frames.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from dronesynth.datagen.annotations import FrameAnnotation

# a drone silhouette fills a fair fraction of its box; below this it's
# probably threshold noise or a sliver of the drone clipped by the frame edge
LOW_FILL_RATIO = 0.15
# boxes smaller than this are legal but small enough to deserve a look
TINY_BOX_AREA = 64


@dataclass(frozen=True)
class QcFlag:
    frame_index: int
    reason: str


@dataclass(frozen=True)
class QcReport:
    run_id: str
    frames: int
    empty_frames: int
    total_boxes: int
    box_area_min: int | None
    box_area_max: int | None
    fill_ratio_min: float | None
    fill_ratio_max: float | None
    flags: tuple[QcFlag, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _frame_flags(annotation: FrameAnnotation) -> list[QcFlag]:
    flags = []
    if len(annotation.boxes) > 1:
        flags.append(QcFlag(annotation.frame_index, f"{len(annotation.boxes)} boxes in frame"))
    for box in annotation.boxes:
        if box.fill_ratio < LOW_FILL_RATIO:
            flags.append(QcFlag(annotation.frame_index, f"low fill ratio {box.fill_ratio}"))
        if box.w * box.h < TINY_BOX_AREA:
            flags.append(QcFlag(annotation.frame_index, f"tiny box {box.w}x{box.h}"))
        touches = (
            box.x == 0
            or box.y == 0
            or box.x + box.w == annotation.width
            or box.y + box.h == annotation.height
        )
        if touches:
            flags.append(QcFlag(annotation.frame_index, "box touches frame edge"))
    return flags


def compute_qc(run_id: str, annotations: list[FrameAnnotation]) -> QcReport:
    boxes = [box for a in annotations for box in a.boxes]
    areas = [box.w * box.h for box in boxes]
    ratios = [box.fill_ratio for box in boxes]
    flags = [flag for a in annotations for flag in _frame_flags(a)]
    return QcReport(
        run_id=run_id,
        frames=len(annotations),
        empty_frames=sum(1 for a in annotations if not a.boxes),
        total_boxes=len(boxes),
        box_area_min=min(areas) if areas else None,
        box_area_max=max(areas) if areas else None,
        fill_ratio_min=min(ratios) if ratios else None,
        fill_ratio_max=max(ratios) if ratios else None,
        flags=tuple(flags),
    )


def write_qc_report(report: QcReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


def render_debug_frame(annotation: FrameAnnotation, normal_path: Path, out_path: Path) -> None:
    """The normal render with its boxes drawn on — human verification material."""
    image = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read normal image: {normal_path}")
    for box in annotation.boxes:
        cv2.rectangle(
            image, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            f"fill {box.fill_ratio:.2f}",
            (box.x, max(box.y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

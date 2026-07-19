"""Discover rendered frames and pair normal/mask renders by frame index.

EasySynth writes both render passes with identical frame numbering (e.g.
``CameraComponent/ColorImage/testSequence.0042.png``), possibly nested in
subfolders. The frame index is the trailing digit group of the filename stem.
Pairing is strict: a frame present on one side but not the other means the
capture is broken, and the whole run is rejected rather than silently
converting a subset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRAME_INDEX_RE = re.compile(r"(\d+)$")

_MAX_LISTED = 5


class PairingError(ValueError):
    """Raised when a capture's frames cannot be discovered or paired."""


@dataclass(frozen=True)
class FramePair:
    index: int
    normal: Path
    mask: Path


def discover_frames(root: Path) -> dict[int, Path]:
    """Map frame index -> file for every PNG under root, searched recursively."""
    if not root.is_dir():
        raise PairingError(f"render directory not found: {root}")
    frames: dict[int, Path] = {}
    for path in sorted(root.rglob("*.png")):
        match = _FRAME_INDEX_RE.search(path.stem)
        if match is None:
            raise PairingError(f"cannot extract a trailing frame index from {path.name}")
        index = int(match.group(1))
        if index in frames:
            raise PairingError(
                f"duplicate frame index {index} under {root}: "
                f"{frames[index].name} and {path.name}"
            )
        frames[index] = path
    if not frames:
        raise PairingError(f"no PNG frames found under {root}")
    return frames


def pair_frames(normal_root: Path, mask_root: Path) -> list[FramePair]:
    """Pair every normal frame with its mask frame; reject incomplete captures."""
    normal = discover_frames(normal_root)
    mask = discover_frames(mask_root)

    missing_mask = sorted(set(normal) - set(mask))
    missing_normal = sorted(set(mask) - set(normal))
    if missing_mask or missing_normal:
        problems = []
        if missing_mask:
            problems.append(f"{len(missing_mask)} frame(s) missing a mask render "
                            f"(e.g. {missing_mask[:_MAX_LISTED]})")
        if missing_normal:
            problems.append(f"{len(missing_normal)} frame(s) missing a normal render "
                            f"(e.g. {missing_normal[:_MAX_LISTED]})")
        raise PairingError("capture is incomplete: " + "; ".join(problems))

    return [FramePair(index=i, normal=normal[i], mask=mask[i]) for i in sorted(normal)]

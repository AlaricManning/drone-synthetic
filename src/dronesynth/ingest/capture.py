"""Register a completed capture as a run.

Validate first, copy second, manifest last:

1. The capture is validated with the same strict pairing conversion uses —
   a broken render is rejected before anything is copied.
2. Frames are copied into the run layout, flattened out of EasySynth's
   nesting and renamed to ``frame_<index>.png``.
3. The manifest is written only after every frame is in place. A run
   directory without a manifest is therefore always debris from a failed
   ingest — never a real run — and a fresh ingest may clear and replace it.
   A run *with* a manifest is immutable and is never touched.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dronesynth.datagen.pairing import FramePair, pair_frames
from dronesynth.ingest.manifest import MANIFEST_FILENAME, RunManifest, write_manifest

_TRAILING_INDEX_RE = re.compile(r"[._-]?\d+$")


class IngestError(ValueError):
    """Raised when a capture cannot be registered as a run."""


@dataclass(frozen=True)
class IngestResult:
    run_dir: Path
    manifest: RunManifest


def sequence_name(pairs: list[FramePair]) -> str:
    """Camera sequence name derived from the filenames: stem minus the index."""
    return _TRAILING_INDEX_RE.sub("", pairs[0].normal.stem)


def ingest_capture(
    *,
    normal_root: Path,
    mask_root: Path,
    run_id: str,
    raw_root: Path,
    captured_at: str,
    ue_map: str,
    drone_model: str,
) -> IngestResult:
    pairs = pair_frames(normal_root, mask_root)

    run_dir = raw_root / run_id
    if run_dir.exists():
        if (run_dir / MANIFEST_FILENAME).is_file():
            raise IngestError(
                f"run {run_id} already exists at {run_dir} — runs are immutable; "
                f"use a new run id"
            )
        # no manifest: by the commit protocol this is debris from a failed
        # ingest, safe to clear and redo
        shutil.rmtree(run_dir)

    manifest = RunManifest(
        run_id=run_id,
        captured_at=captured_at,
        frame_count=len(pairs),
        ue_map=ue_map,
        drone_model=drone_model,
        camera_sequence=sequence_name(pairs),
    )

    for side in ("normal", "mask"):
        (run_dir / side).mkdir(parents=True)
    for pair in pairs:
        name = f"frame_{pair.index:06d}.png"
        shutil.copy2(pair.normal, run_dir / "normal" / name)
        shutil.copy2(pair.mask, run_dir / "mask" / name)

    write_manifest(manifest, run_dir)
    return IngestResult(run_dir=run_dir, manifest=manifest)

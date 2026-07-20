"""Register a completed capture as a run, locally or in S3.

Validate first, upload second, manifest last:

1. The capture is validated with the same strict pairing conversion uses —
   a broken render is rejected before anything is uploaded.
2. Frames go into the run layout, flattened out of EasySynth's nesting and
   renamed to ``frame_<index>.png``. Re-uploading over debris from a failed
   ingest is safe: the keys are deterministic, so retries overwrite.
3. The manifest is written last, with if-absent semantics: two ingests of
   the same run id cannot both succeed, which is what makes runs immutable —
   even on S3, where the put-only ingest identity cannot look before it
   writes.

On local storage, debris from a failed ingest is cleared before the retry.
On S3 the ingest identity cannot delete (by design), so stale objects from a
failed ingest of a *different-shaped* capture can linger; convert's
manifest/frame-count cross-check catches that, and cleanup is an admin task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from dronesynth.datagen.pairing import FramePair, pair_frames
from dronesynth.ingest.manifest import MANIFEST_FILENAME, RunManifest
from dronesynth.storage import StorageKeyExists, StorageNotPermitted, storage_for

_TRAILING_INDEX_RE = re.compile(r"[._-]?\d+$")


class IngestError(ValueError):
    """Raised when a capture cannot be registered as a run."""


@dataclass(frozen=True)
class IngestResult:
    location: str  # human-readable: where the run landed
    manifest: RunManifest


def sequence_name(pairs: list[FramePair]) -> str:
    """Camera sequence name derived from the filenames: stem minus the index."""
    return _TRAILING_INDEX_RE.sub("", pairs[0].normal.stem)


def ingest_capture(
    *,
    normal_root: Path,
    mask_root: Path,
    run_id: str,
    raw_root: str,
    captured_at: str,
    ue_map: str,
    drone_model: str,
) -> IngestResult:
    pairs = pair_frames(normal_root, mask_root)
    storage = storage_for(str(raw_root))
    manifest_key = f"{run_id}/{MANIFEST_FILENAME}"

    # best-effort early check; put-only S3 credentials can't look, and the
    # if-absent manifest write below enforces immutability regardless
    try:
        if storage.exists(manifest_key):
            raise IngestError(
                f"run {run_id} already exists at {storage.describe(run_id)} — "
                f"runs are immutable; use a new run id"
            )
        storage.delete_prefix(run_id)  # clear debris from a failed ingest
    except StorageNotPermitted:
        pass

    manifest = RunManifest(
        run_id=run_id,
        captured_at=captured_at,
        frame_count=len(pairs),
        ue_map=ue_map,
        drone_model=drone_model,
        camera_sequence=sequence_name(pairs),
    )

    for pair in pairs:
        name = f"frame_{pair.index:06d}.png"
        storage.put_file(pair.normal, f"{run_id}/normal/{name}")
        storage.put_file(pair.mask, f"{run_id}/mask/{name}")

    try:
        storage.write_text_if_absent(manifest_key, json.dumps(manifest.to_dict(), indent=2))
    except StorageKeyExists as exc:
        raise IngestError(
            f"run {run_id} already exists at {storage.describe(run_id)} — "
            f"runs are immutable; use a new run id"
        ) from exc

    return IngestResult(location=storage.describe(run_id), manifest=manifest)

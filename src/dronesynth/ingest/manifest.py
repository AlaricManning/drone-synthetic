"""The run manifest: what makes a capture a run.

A run directory without a manifest is incomplete by definition — ingest
writes the manifest only after every frame is in place, so a crashed or
half-finished ingest can never be mistaken for a real run. The manifest is
also the run's provenance record: what was rendered, from what scene, and
(for future captures) under which domain-randomization parameters.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

# run ids become directory names and S3 key prefixes; keep them boring
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ManifestError(ValueError):
    """Raised when a manifest is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    captured_at: str  # ISO date of the render session
    frame_count: int
    ue_map: str
    drone_model: str
    camera_sequence: str
    # reserved for in-Unreal domain randomization; empty until that exists
    randomization: dict = field(default_factory=dict)
    seed: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _RUN_ID_RE.match(self.run_id):
            raise ManifestError(
                f"run_id must be lowercase alphanumeric/_/- (got {self.run_id!r})"
            )
        try:
            date.fromisoformat(self.captured_at)
        except ValueError as exc:
            raise ManifestError(
                f"captured_at must be an ISO date like 2026-07-19 (got {self.captured_at!r})"
            ) from exc
        if self.frame_count <= 0:
            raise ManifestError(f"frame_count must be positive (got {self.frame_count})")
        for name in ("ue_map", "drone_model", "camera_sequence"):
            if not getattr(self, name).strip():
                raise ManifestError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunManifest:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported manifest schema_version {version!r} (expected {SCHEMA_VERSION})"
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise ManifestError(f"malformed manifest: {exc}") from exc


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    path = run_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def read_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise ManifestError(
            f"no {MANIFEST_FILENAME} in {run_dir} — the run is incomplete or not a run"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"expected a JSON object in {path}")
    return RunManifest.from_dict(data)

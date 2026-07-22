"""Orchestrates one run's conversion: paired renders -> dataset + QC.

The unit of work the Batch job executes. All three storage roots (raw,
dataset, qc) may independently be local paths or s3:// URIs:

- the run is *localized* first — downloaded to a working directory when raw
  storage is remote, used in place when it is already a local directory;
- conversion then runs entirely against local files, exactly as in
  development;
- outputs are *published* through the storage layer — written in place when
  the destination is local, uploaded when it is S3.

Deterministic: (run frames, config) fully determine every output byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from dronesynth.config import ConvertConfig
from dronesynth.datagen.annotations import annotate_frame, write_annotations
from dronesynth.datagen.pairing import pair_frames
from dronesynth.datagen.qc import QcReport, compute_qc, render_debug_frame, write_qc_report
from dronesynth.datagen.split import split_runs
from dronesynth.datagen.yolo import ExportItem, export_yolo
from dronesynth.ingest.manifest import MANIFEST_FILENAME, ManifestError, RunManifest
from dronesynth.storage import LocalStorage, Storage, StorageKeyMissing, storage_for


class ConvertError(ValueError):
    """Raised when a run cannot be converted."""


@dataclass(frozen=True)
class ConvertResult:
    report: QcReport
    dataset_location: str
    qc_location: str


def _localize_run(raw: Storage, run_id: str, workdir: Path) -> Path:
    """Make the run's frames available as a local directory."""
    if isinstance(raw, LocalStorage):
        return raw.root / run_id
    run_dir = workdir / "raw" / run_id
    for key in raw.list_keys(run_id):
        raw.get_file(key, workdir / "raw" / key)
    return run_dir


def _publish(src_dir: Path, dest: Storage, prefix: str) -> None:
    """Upload a finished local output tree; no-op when it was written in place."""
    if isinstance(dest, LocalStorage):
        return
    for path in sorted(src_dir.rglob("*")):
        if path.is_file():
            dest.put_file(path, f"{prefix}/{path.relative_to(src_dir).as_posix()}")


def _out_dir(storage: Storage, prefix: str, workdir: Path, label: str) -> Path:
    """Where conversion writes: the final location if local, a staging dir if S3."""
    if isinstance(storage, LocalStorage):
        return storage.root / prefix
    return workdir / label


def convert_run(
    run_id: str,
    config: ConvertConfig,
    dataset_version: str,
) -> ConvertResult:
    if len(config.class_map) != 1:
        raise NotImplementedError(
            "multi-class masks are not supported yet; class_map must have exactly one entry"
        )
    class_id = next(iter(config.class_map))

    raw = storage_for(config.storage.raw_root)
    dataset_storage = storage_for(config.storage.dataset_root)
    qc_storage = storage_for(config.storage.qc_root)

    try:
        manifest_text = raw.read_text(f"{run_id}/{MANIFEST_FILENAME}")
    except StorageKeyMissing as exc:
        raise ConvertError(
            f"no {MANIFEST_FILENAME} at {raw.describe(run_id)} — "
            f"the run is incomplete or not a run"
        ) from exc
    try:
        manifest = RunManifest.from_dict(json.loads(manifest_text))
    except (json.JSONDecodeError, ManifestError) as exc:
        raise ConvertError(f"invalid manifest for run {run_id}: {exc}") from exc

    with TemporaryDirectory(prefix=f"dronesynth-{run_id}-") as tmp:
        workdir = Path(tmp)
        run_dir = _localize_run(raw, run_id, workdir)

        pairs = pair_frames(run_dir / "normal", run_dir / "mask")
        if len(pairs) != manifest.frame_count:
            raise ConvertError(
                f"run {run_id} has {len(pairs)} frame pairs in storage but its manifest "
                f"says {manifest.frame_count} — the run is corrupt"
            )

        annotations = [
            annotate_frame(
                pair,
                threshold=config.mask.threshold,
                min_box_area=config.mask.min_box_area,
                class_id=class_id,
            )
            for pair in pairs
        ]

        dataset_dir = _out_dir(dataset_storage, dataset_version, workdir, "dataset")
        write_annotations(annotations, dataset_dir / "annotations" / f"{run_id}.json")

        assignments = split_runs([run_id], config.split.val_runs)
        items = [
            ExportItem(run_id=run_id, annotation=annotation, image_path=pair.normal)
            for pair, annotation in zip(pairs, annotations, strict=True)
        ]
        export_yolo(items, dataset_dir / "yolo", config.class_map, assignments)

        qc_dir = _out_dir(qc_storage, run_id, workdir, "qc")
        report = compute_qc(run_id, annotations)
        write_qc_report(report, qc_dir / "report.json")
        for pair, annotation in zip(pairs, annotations, strict=True):
            out = qc_dir / "debug" / f"{run_id}_{annotation.frame_index:06d}.png"
            render_debug_frame(annotation, pair.normal, out)

        _publish(dataset_dir, dataset_storage, dataset_version)
        _publish(qc_dir, qc_storage, run_id)

    return ConvertResult(
        report=report,
        dataset_location=dataset_storage.describe(dataset_version),
        qc_location=qc_storage.describe(run_id),
    )

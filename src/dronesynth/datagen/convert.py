"""Orchestrates one run's conversion: paired renders -> dataset + QC.

The unit of work the Batch job will eventually execute. Deterministic:
(run frames, config) fully determine every output byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dronesynth.config import ConvertConfig
from dronesynth.datagen.annotations import annotate_frame, write_annotations
from dronesynth.datagen.pairing import pair_frames
from dronesynth.datagen.qc import QcReport, compute_qc, render_debug_frame, write_qc_report
from dronesynth.datagen.split import split_runs
from dronesynth.datagen.yolo import ExportItem, export_yolo
from dronesynth.ingest.manifest import read_manifest


class ConvertError(ValueError):
    """Raised when a run cannot be converted."""


@dataclass(frozen=True)
class ConvertResult:
    report: QcReport
    dataset_dir: Path
    qc_dir: Path


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

    run_dir = Path(config.storage.raw_root) / run_id
    manifest = read_manifest(run_dir)  # no manifest -> not a run, refuse loudly
    pairs = pair_frames(run_dir / "normal", run_dir / "mask")
    if len(pairs) != manifest.frame_count:
        raise ConvertError(
            f"run {run_id} has {len(pairs)} frame pairs on disk but its manifest "
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

    dataset_dir = Path(config.storage.dataset_root) / dataset_version
    write_annotations(annotations, dataset_dir / "annotations" / f"{run_id}.json")

    assignments = split_runs([run_id], config.split.val_runs)
    items = [
        ExportItem(run_id=run_id, annotation=annotation, image_path=pair.normal)
        for pair, annotation in zip(pairs, annotations, strict=True)
    ]
    export_yolo(items, dataset_dir / "yolo", config.class_map, assignments)

    qc_dir = Path(config.storage.qc_root) / run_id
    report = compute_qc(run_id, annotations)
    write_qc_report(report, qc_dir / "report.json")
    for pair, annotation in zip(pairs, annotations, strict=True):
        out = qc_dir / "debug" / f"{run_id}_{annotation.frame_index:06d}.png"
        render_debug_frame(annotation, pair.normal, out)

    return ConvertResult(report=report, dataset_dir=dataset_dir, qc_dir=qc_dir)

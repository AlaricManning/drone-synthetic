"""Assembling many converted runs into one trainable, auditable dataset.

Conversion is per-run and answers "what is in these frames". It cannot answer
"which runs does this dataset train on and which does it validate against",
because that question does not exist until something decides which runs the
dataset contains. Nothing did, which is why the corpus finished as 50 separate
single-run versions that no one could train on, and why ``split.val_runs`` broke
every conversion it was set on. See docs/plans/dataset-build.md.

A build reads the canonical annotations the converter already wrote, copies the
normal frames those annotations label, writes labels and a dataset descriptor,
and records what it did. It does not re-threshold a single mask: labels are a
pure function of annotations, so a build cannot disagree with the QC report its
inputs came from, and re-splitting recopies files while recomputing nothing.

Frames come from ``raw/`` rather than from the per-run YOLO exports, which are
duplicates of them and are due for removal once this path is proven. Copies go
key-to-key through Storage.copy_from, so on S3 the roughly 9 GB never leaves the
bucket.

Complete by construction, exactly as a run is: frames, then labels, then the
descriptor, then the manifest last and only if absent. A version without a
manifest is debris from a failed build; a version with one is immutable, and
rebuilding it is an error for the same reason re-ingesting a run id is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from dronesynth.config import BuildConfig
from dronesynth.datagen.annotations import FrameAnnotation, parse_annotations
from dronesynth.datagen.split import split_runs
from dronesynth.datagen.yolo import dataset_yaml_text, yolo_label_lines
from dronesynth.ingest.manifest import MANIFEST_FILENAME, ManifestError, RunManifest
from dronesynth.provenance import PROVENANCE_SUFFIX, RunProvenance, converter_stamp
from dronesynth.storage import Storage, StorageKeyExists, StorageKeyMissing, storage_for

DATASET_MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

# What the Lambda names each per-run conversion. The build reads its inputs from
# there rather than being told, so a build config cannot name a run and then
# point at some other run's annotations.
AUTO_VERSION = "auto-{run_id}"


class BuildError(ValueError):
    """Raised when a dataset cannot be assembled from the requested runs."""


@dataclass(frozen=True)
class RunInput:
    """One run's contribution, gathered before anything is written."""

    run_id: str
    source_version: str
    manifest: RunManifest
    provenance: RunProvenance
    annotations: tuple[FrameAnnotation, ...]

    @property
    def boxes(self) -> int:
        return sum(len(a.boxes) for a in self.annotations)

    @property
    def scene_identity(self) -> str:
        """What makes this run's *scene* distinct, as opposed to this render of it.

        Run ids are timestamps, so they are unique per render rather than per
        scene: when the two render batches collided, 25 seeds rendered twice and
        produced pairs with different ids and byte-identical trajectories. A
        run-level split would have put one of each pair in train and its twin in
        val and reported nothing wrong.

        Seed alone is too strict — the same seed under a different config or a
        different renderer build genuinely is a different scene — so the identity
        is the seed together with everything that decides what it means.
        """
        payload = json.dumps(
            {
                "seed": self.manifest.seed,
                "randomization": self.manifest.randomization,
                "ue_map": self.manifest.ue_map,
                "drone_model": self.manifest.drone_model,
                "generator": self.manifest.generator.get("commit"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe_scene(self) -> str:
        seed = self.manifest.seed
        return f"seed {seed}" if seed is not None else f"scene {self.scene_identity}"


@dataclass(frozen=True)
class DatasetManifest:
    """The audit record: what this dataset is, and everything that made it.

    Embeds each input run's metadata rather than referencing run ids, because
    inputs are mutable and the record cannot depend on them. Pruning the 25
    duplicate runs deleted 7625 objects including their manifests; a dataset
    that merely pointed at run ids would have lost the ability to explain
    itself the moment that happened.
    """

    version: str
    built_at: str
    builder: dict[str, Any] = field(default_factory=dict)
    conversion: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    runs: tuple[dict[str, Any], ...] = ()
    totals: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "built_at": self.built_at,
            "schema_version": self.schema_version,
            "builder": self.builder,
            "conversion": self.conversion,
            "split": self.split,
            "totals": self.totals,
            "runs": list(self.runs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise BuildError(
                f"unsupported dataset manifest schema_version {version!r} "
                f"(expected {SCHEMA_VERSION})"
            )
        return cls(
            version=data["version"],
            built_at=data["built_at"],
            builder=data.get("builder", {}),
            conversion=data.get("conversion", {}),
            split=data.get("split", {}),
            runs=tuple(data.get("runs", ())),
            totals=data.get("totals", {}),
        )


@dataclass(frozen=True)
class BuildResult:
    manifest: DatasetManifest
    location: str


def _gather_run(run_id: str, raw: Storage, datasets: Storage) -> RunInput:
    source_version = AUTO_VERSION.format(run_id=run_id)

    try:
        manifest = RunManifest.from_dict(json.loads(raw.read_text(f"{run_id}/{MANIFEST_FILENAME}")))
    except StorageKeyMissing as exc:
        raise BuildError(
            f"run {run_id} has no {MANIFEST_FILENAME} at {raw.describe(run_id)} — "
            f"it was never ingested, or it has been pruned"
        ) from exc
    except (json.JSONDecodeError, ManifestError) as exc:
        raise BuildError(f"run {run_id} has an invalid manifest: {exc}") from exc

    annotations_key = f"{source_version}/annotations/{run_id}.json"
    try:
        annotations = parse_annotations(datasets.read_text(annotations_key))
    except StorageKeyMissing as exc:
        raise BuildError(
            f"run {run_id} has no annotations at {datasets.describe(annotations_key)} — "
            f"convert it before building"
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BuildError(f"run {run_id} has invalid annotations: {exc}") from exc

    provenance_key = f"{source_version}/annotations/{run_id}{PROVENANCE_SUFFIX}"
    try:
        provenance = RunProvenance.from_dict(json.loads(datasets.read_text(provenance_key)))
    except StorageKeyMissing as exc:
        # Deliberately fatal rather than a warning. A dataset whose labels
        # cannot be attributed to a converter build and a mask config does not
        # meet the requirement this whole path exists for; reconverting the run
        # is cheap and gives a clean answer.
        raise BuildError(
            f"run {run_id} has no converter provenance at "
            f"{datasets.describe(provenance_key)} — it was converted before "
            f"provenance was recorded, so reconvert it before building"
        ) from exc
    except (json.JSONDecodeError, KeyError) as exc:
        raise BuildError(f"run {run_id} has invalid provenance: {exc}") from exc

    if len(annotations) != manifest.frame_count:
        raise BuildError(
            f"run {run_id} has {len(annotations)} annotated frames but its manifest says "
            f"{manifest.frame_count} — the conversion is stale or the run is corrupt"
        )

    return RunInput(
        run_id=run_id,
        source_version=source_version,
        manifest=manifest,
        provenance=provenance,
        annotations=tuple(annotations),
    )


def _require_comparable(inputs: list[RunInput]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One converter build and one mask config across all inputs, or refuse.

    Labels produced under a different threshold are not the same kind of label,
    and averaging over both silently is how a dataset ends up meaning nothing in
    particular. This is a live hazard rather than a theoretical one: runs
    converted before the threshold moved from 12 to 32 sit in the same bucket as
    the corpus and look identical from the outside.
    """
    by_conversion: dict[str, list[str]] = {}
    by_converter: dict[str, list[str]] = {}
    for run in inputs:
        by_conversion.setdefault(json.dumps(run.provenance.conversion, sort_keys=True), []).append(
            run.run_id
        )
        by_converter.setdefault(json.dumps(run.provenance.converter, sort_keys=True), []).append(
            run.run_id
        )

    for label, groups in (("conversion config", by_conversion), ("converter build", by_converter)):
        if len(groups) > 1:
            detail = "; ".join(
                f"{blob} <- {len(runs)} run(s) e.g. {runs[0]}"
                for blob, runs in sorted(groups.items())
            )
            raise BuildError(
                f"input runs were converted under {len(groups)} different "
                f"{label}s, so their labels are not comparable: {detail}"
            )

    return inputs[0].provenance.conversion, inputs[0].provenance.converter


def _require_no_scene_leak(inputs: list[RunInput], assignments: dict[str, str]) -> None:
    by_scene: dict[str, dict[str, list[str]]] = {}
    for run in inputs:
        sides = by_scene.setdefault(run.scene_identity, {})
        sides.setdefault(assignments[run.run_id], []).append(run.run_id)

    leaks = [sides for sides in by_scene.values() if len(sides) > 1]
    if not leaks:
        return

    detail = "; ".join(
        ", ".join(f"{side}={sorted(sides[side])}" for side in sorted(sides)) for sides in leaks
    )
    raise BuildError(
        f"{len(leaks)} scene(s) appear on both sides of the split, which leaks "
        f"validation data even though the run ids differ: {detail}"
    )


def _run_record(run: RunInput, subset: str) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "source_version": run.source_version,
        "subset": subset,
        "seed": run.manifest.seed,
        "captured_at": run.manifest.captured_at,
        "ue_map": run.manifest.ue_map,
        "drone_model": run.manifest.drone_model,
        "camera_sequence": run.manifest.camera_sequence,
        "randomization": run.manifest.randomization,
        "generator": run.manifest.generator,
        "converter": run.provenance.converter,
        "converted_at": run.provenance.converted_at,
        "scene_identity": run.scene_identity,
        "frames": len(run.annotations),
        "boxes": run.boxes,
    }


def build_dataset(version: str, config: BuildConfig) -> BuildResult:
    """Assemble one dataset version from the runs named in config."""
    raw = storage_for(config.storage.raw_root)
    datasets = storage_for(config.storage.dataset_root)

    manifest_key = f"{version}/{DATASET_MANIFEST_FILENAME}"
    if datasets.exists(manifest_key):
        raise BuildError(
            f"dataset version {version} already exists at {datasets.describe(version)} — "
            f"versions are immutable; build a new one"
        )

    inputs = [_gather_run(run_id, raw, datasets) for run_id in config.runs]
    conversion, converter = _require_comparable(inputs)

    # split_runs finally has the multi-run caller it was written for, so its
    # rejection of unknown val runs starts protecting something: a typo in the
    # hold-out list would otherwise yield a silently empty validation set.
    assignments = split_runs(config.runs, config.split.val_runs)
    _require_no_scene_leak(inputs, assignments)

    class_map = {int(k): v for k, v in conversion.get("class_map", {}).items()}
    if not class_map:
        raise BuildError(
            "input runs record no class_map in their provenance, so no dataset "
            "descriptor can be written"
        )

    ordered = sorted(inputs, key=lambda r: r.run_id)
    counts = {"train": 0, "val": 0}
    for run in ordered:
        subset = assignments[run.run_id]
        for annotation in run.annotations:
            stem = f"{run.run_id}_{annotation.frame_index:06d}"
            suffix = PurePosixPath(annotation.normal).suffix
            datasets.copy_from(
                raw,
                f"{run.run_id}/normal/{annotation.normal}",
                f"{version}/yolo/images/{subset}/{stem}{suffix}",
            )
            lines = yolo_label_lines(annotation)
            datasets.write_text(
                f"{version}/yolo/labels/{subset}/{stem}.txt",
                "\n".join(lines) + "\n" if lines else "",
            )
            counts[subset] += 1

    datasets.write_text(f"{version}/yolo/dataset.yaml", dataset_yaml_text(class_map))

    train = sorted(r for r, s in assignments.items() if s == "train")
    val = sorted(r for r, s in assignments.items() if s == "val")
    manifest = DatasetManifest(
        version=version,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Same repo as the converter, so the same stamp answers "which build
        # assembled this" -- for a build run inside the conversion image, the
        # commit baked in there; for a local build, the checkout's.
        builder=converter_stamp(),
        conversion=conversion,
        split={"mode": config.split.mode, "train": train, "val": val},
        runs=tuple(_run_record(r, assignments[r.run_id]) for r in ordered),
        totals={
            "runs": len(inputs),
            "frames": sum(len(r.annotations) for r in inputs),
            "boxes": sum(r.boxes for r in inputs),
            "train_runs": len(train),
            "val_runs": len(val),
            "train_frames": counts["train"],
            "val_frames": counts["val"],
        },
    )

    # Manifest last and only if absent, the protocol ingest uses for runs. Two
    # builds racing on one version both write frames; only one gets the manifest,
    # and the loser's partial version is identifiable as debris.
    try:
        datasets.write_text_if_absent(
            manifest_key, json.dumps(manifest.to_dict(), indent=2)
        )
    except StorageKeyExists as exc:
        raise BuildError(
            f"dataset version {version} was completed by another build while this "
            f"one was running; this build's frames are debris"
        ) from exc

    return BuildResult(manifest=manifest, location=datasets.describe(version))

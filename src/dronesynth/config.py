"""Typed loading and validation of the conversion config.

A dataset version is fully determined by (input runs, config), so every knob
that affects dataset content must come through here — nothing content-affecting
is hardcoded or read from the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class MaskConfig:
    threshold: int
    min_box_area: int


@dataclass(frozen=True)
class SplitConfig:
    mode: str
    val_runs: tuple[str, ...]


@dataclass(frozen=True)
class StorageConfig:
    raw_root: str
    dataset_root: str
    qc_root: str


@dataclass(frozen=True)
class ConvertConfig:
    class_map: dict[int, str]
    mask: MaskConfig
    split: SplitConfig
    storage: StorageConfig


@dataclass(frozen=True)
class BuildStorageConfig:
    """A build reads raw frames and converted annotations; it writes no QC."""

    raw_root: str
    dataset_root: str


@dataclass(frozen=True)
class BuildConfig:
    """Which runs a dataset version is assembled from, and how it is split.

    Deliberately holds the run list rather than taking it on the command line.
    Which runs a dataset trains on is the most consequential decision in a
    build, and this way it lives in a reviewable file that git tracks, so a
    version is determined by (this config, version name) with nothing
    important passed in transiently. Expect one of these per dataset version.

    No class_map and no mask settings: both are properties of the conversion
    that produced the annotations, read back from the runs' provenance rather
    than restated here where they could contradict it.
    """

    runs: tuple[str, ...]
    split: SplitConfig
    storage: BuildStorageConfig


def _require(section: dict, key: str, context: str):
    if key not in section:
        raise ConfigError(f"missing required key '{key}' in {context}")
    return section[key]


def _load_yaml_mapping(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return raw


def _load_split(raw: dict, context: str) -> SplitConfig:
    split_raw = _require(raw, "split", context)
    mode = _require(split_raw, "mode", "split")
    if mode != "by_run":
        raise ConfigError(f"split.mode must be 'by_run' (frame-level splits leak), got {mode!r}")
    val_runs = split_raw.get("val_runs") or []
    if not isinstance(val_runs, list) or not all(isinstance(r, str) for r in val_runs):
        raise ConfigError("split.val_runs must be a list of run ids")
    return SplitConfig(mode=mode, val_runs=tuple(val_runs))


def load_convert_config(path: Path) -> ConvertConfig:
    raw = _load_yaml_mapping(path)

    class_map_raw = _require(raw, "class_map", str(path))
    if not isinstance(class_map_raw, dict) or not class_map_raw:
        raise ConfigError("class_map must be a non-empty mapping of class id -> name")
    class_map: dict[int, str] = {}
    for key, name in class_map_raw.items():
        if not isinstance(key, int) or key < 0:
            raise ConfigError(f"class_map ids must be non-negative integers, got {key!r}")
        class_map[key] = str(name)

    mask_raw = _require(raw, "mask", str(path))
    threshold = _require(mask_raw, "threshold", "mask")
    min_box_area = _require(mask_raw, "min_box_area", "mask")
    if not isinstance(threshold, int) or not 0 <= threshold <= 254:
        raise ConfigError(f"mask.threshold must be an integer in [0, 254], got {threshold!r}")
    if not isinstance(min_box_area, int) or min_box_area < 0:
        raise ConfigError(f"mask.min_box_area must be a non-negative integer, got {min_box_area!r}")

    split = _load_split(raw, str(path))

    storage_raw = _require(raw, "storage", str(path))
    storage = StorageConfig(
        raw_root=str(_require(storage_raw, "raw_root", "storage")),
        dataset_root=str(_require(storage_raw, "dataset_root", "storage")),
        qc_root=str(_require(storage_raw, "qc_root", "storage")),
    )

    return ConvertConfig(
        class_map=class_map,
        mask=MaskConfig(threshold=threshold, min_box_area=min_box_area),
        split=split,
        storage=storage,
    )


def load_build_config(path: Path) -> BuildConfig:
    raw = _load_yaml_mapping(path)

    runs_raw = _require(raw, "runs", str(path))
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ConfigError("runs must be a non-empty list of run ids")
    if not all(isinstance(r, str) and r.strip() for r in runs_raw):
        raise ConfigError("runs must be a list of non-empty run ids")
    runs = [r.strip() for r in runs_raw]
    duplicates = sorted({r for r in runs if runs.count(r) > 1})
    if duplicates:
        # Silently deduplicating would weight those runs twice in training and
        # make the manifest's frame totals disagree with the frames present.
        raise ConfigError(f"runs contains duplicates: {duplicates}")

    storage_raw = _require(raw, "storage", str(path))
    storage = BuildStorageConfig(
        raw_root=str(_require(storage_raw, "raw_root", "storage")),
        dataset_root=str(_require(storage_raw, "dataset_root", "storage")),
    )

    return BuildConfig(
        runs=tuple(runs),
        split=_load_split(raw, str(path)),
        storage=storage,
    )

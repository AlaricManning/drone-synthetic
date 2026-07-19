import json

import pytest

from dronesynth.ingest.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    RunManifest,
    read_manifest,
    write_manifest,
)


def manifest(**overrides):
    fields = dict(
        run_id="run_0001",
        captured_at="2026-07-19",
        frame_count=150,
        ue_map="SkyTestMap",
        drone_model="Quadcopter_A",
        camera_sequence="testSequence",
    )
    fields.update(overrides)
    return RunManifest(**fields)


def test_round_trip(tmp_path):
    original = manifest(randomization={"sun_angle": [0, 90]}, seed=42)
    write_manifest(original, tmp_path)
    assert read_manifest(tmp_path) == original


def test_randomization_defaults_empty():
    m = manifest()
    assert m.randomization == {}
    assert m.seed is None


def test_bad_run_id_rejected():
    for bad in ("Run_0001", "run/0001", "", "../etc"):
        with pytest.raises(ManifestError, match="run_id"):
            manifest(run_id=bad)


def test_bad_date_rejected():
    with pytest.raises(ManifestError, match="ISO date"):
        manifest(captured_at="19/07/2026")


def test_bad_frame_count_rejected():
    with pytest.raises(ManifestError, match="frame_count"):
        manifest(frame_count=0)


def test_empty_metadata_rejected():
    with pytest.raises(ManifestError, match="ue_map"):
        manifest(ue_map="  ")


def test_missing_manifest(tmp_path):
    with pytest.raises(ManifestError, match="incomplete or not a run"):
        read_manifest(tmp_path)


def test_corrupt_manifest(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text("{not json")
    with pytest.raises(ManifestError, match="invalid JSON"):
        read_manifest(tmp_path)


def test_unknown_schema_version_rejected(tmp_path):
    data = manifest().to_dict()
    data["schema_version"] = 99
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="schema_version"):
        read_manifest(tmp_path)


def test_unknown_field_rejected(tmp_path):
    data = manifest().to_dict()
    data["surprise"] = True
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="malformed"):
        read_manifest(tmp_path)

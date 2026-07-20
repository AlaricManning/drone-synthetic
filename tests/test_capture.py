import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from dronesynth.datagen.pairing import PairingError
from dronesynth.ingest.capture import IngestError, ingest_capture
from dronesynth.ingest.manifest import MANIFEST_FILENAME, read_manifest
from dronesynth.storage import backends as backends_module


def write_capture(root, indices, sides=("normal", "mask")):
    for side in sides:
        directory = root / side / "CameraComponent" / "ColorImage"
        directory.mkdir(parents=True, exist_ok=True)
        for i in indices:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(directory / f"testSequence.{i:04d}.png"), image)


def ingest(tmp_path, **overrides):
    kwargs = dict(
        normal_root=tmp_path / "normal",
        mask_root=tmp_path / "mask",
        run_id="run_0001",
        raw_root=str(tmp_path / "raw"),
        captured_at="2026-07-19",
        ue_map="SkyTestMap",
        drone_model="Quadcopter_A",
    )
    kwargs.update(overrides)
    return ingest_capture(**kwargs)


def test_ingest_layout_and_manifest(tmp_path):
    write_capture(tmp_path, range(3))
    result = ingest(tmp_path)

    run_dir = tmp_path / "raw" / "run_0001"
    assert result.location == str(run_dir)
    for side in ("normal", "mask"):
        names = sorted(p.name for p in (run_dir / side).iterdir())
        assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]

    manifest = read_manifest(run_dir)
    assert manifest.frame_count == 3
    assert manifest.camera_sequence == "testSequence"
    assert result.manifest == manifest


def test_broken_capture_rejected_before_copying(tmp_path):
    write_capture(tmp_path, range(3))
    write_capture(tmp_path, [3], sides=("normal",))  # frame 3 has no mask
    with pytest.raises(PairingError, match="incomplete"):
        ingest(tmp_path)
    assert not (tmp_path / "raw").exists()


def test_existing_run_with_manifest_is_immutable(tmp_path):
    write_capture(tmp_path, range(3))
    ingest(tmp_path)
    with pytest.raises(IngestError, match="immutable"):
        ingest(tmp_path)


def test_debris_without_manifest_is_replaced(tmp_path):
    write_capture(tmp_path, range(3))
    debris = tmp_path / "raw" / "run_0001" / "normal"
    debris.mkdir(parents=True)
    (debris / "frame_999999.png").touch()

    result = ingest(tmp_path)
    run_dir = tmp_path / "raw" / "run_0001"
    assert result.location == str(run_dir)
    names = sorted(p.name for p in (run_dir / "normal").iterdir())
    assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]


def test_failed_ingest_leaves_no_manifest(tmp_path, monkeypatch):
    write_capture(tmp_path, range(3))
    real_copy = backends_module.copy2
    calls = {"n": 0}

    def flaky_copy(src, dst, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk full")
        return real_copy(src, dst, **kwargs)

    monkeypatch.setattr(backends_module, "copy2", flaky_copy)
    with pytest.raises(OSError):
        ingest(tmp_path)

    run_dir = tmp_path / "raw" / "run_0001"
    assert run_dir.exists()
    assert not (run_dir / MANIFEST_FILENAME).exists()  # debris, not a run

    # and a retry self-heals: clears the debris and completes
    monkeypatch.setattr(backends_module, "copy2", real_copy)
    result = ingest(tmp_path)
    assert (run_dir / MANIFEST_FILENAME).is_file()
    assert result.manifest.frame_count == 3


def test_ingest_to_s3(tmp_path):
    write_capture(tmp_path, range(3))
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="synth-bucket")

        result = ingest(tmp_path, raw_root="s3://synth-bucket/raw")
        assert result.location == "s3://synth-bucket/raw/run_0001"

        keys = sorted(
            entry["Key"]
            for entry in client.list_objects_v2(Bucket="synth-bucket")["Contents"]
        )
        assert keys == [
            "raw/run_0001/manifest.json",
            "raw/run_0001/mask/frame_000000.png",
            "raw/run_0001/mask/frame_000001.png",
            "raw/run_0001/mask/frame_000002.png",
            "raw/run_0001/normal/frame_000000.png",
            "raw/run_0001/normal/frame_000001.png",
            "raw/run_0001/normal/frame_000002.png",
        ]

        # re-ingesting the same run id must fail even though the S3 ingest
        # identity can't check existence first — the conditional manifest
        # write is the enforcement
        with pytest.raises(IngestError, match="immutable"):
            ingest(tmp_path, raw_root="s3://synth-bucket/raw")

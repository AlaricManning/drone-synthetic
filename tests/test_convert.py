import json

import cv2
import numpy as np
import pytest

from dronesynth.config import load_convert_config
from dronesynth.datagen.convert import ConvertError, convert_run
from dronesynth.ingest.capture import ingest_capture
from dronesynth.ingest.manifest import ManifestError

CONFIG = """\
class_map:
  0: drone
mask:
  threshold: 12
  min_box_area: 16
split:
  mode: by_run
  val_runs: []
storage:
  raw_root: {root}/raw
  dataset_root: {root}/datasets
  qc_root: {root}/qc
"""


def write_capture(root, frames):
    """frames: dict index -> blob (y0, y1, x0, x1) or None for a drone-less frame."""
    for side in ("normal", "mask"):
        (root / side).mkdir(parents=True)
    for index, blob in frames.items():
        normal = np.full((48, 64, 3), 90, dtype=np.uint8)
        mask = np.zeros((48, 64, 4), dtype=np.uint8)  # BGRA like EasySynth
        mask[:, :, 3] = 255
        if blob is not None:
            y0, y1, x0, x1 = blob
            mask[y0:y1, x0:x1, :3] = 255
        for side, image in (("normal", normal), ("mask", mask)):
            cv2.imwrite(str(root / side / f"seq.{index:04d}.png"), image)


def registered_config(tmp_path, frames):
    write_capture(tmp_path / "capture", frames)
    config_path = tmp_path / "convert.yaml"
    config_path.write_text(CONFIG.format(root=tmp_path))
    config = load_convert_config(config_path)
    ingest_capture(
        normal_root=tmp_path / "capture" / "normal",
        mask_root=tmp_path / "capture" / "mask",
        run_id="run_0001",
        raw_root=tmp_path / "raw",
        captured_at="2026-07-19",
        ue_map="SkyTestMap",
        drone_model="Quadcopter_A",
    )
    return config


def test_convert_registered_run_end_to_end(tmp_path):
    config = registered_config(tmp_path, {0: (10, 30, 5, 15), 1: None, 2: (20, 40, 30, 50)})

    result = convert_run(run_id="run_0001", config=config, dataset_version="v001")

    assert result.report.frames == 3
    assert result.report.empty_frames == 1
    assert result.report.total_boxes == 2

    dataset = tmp_path / "datasets" / "v001"
    annotations = json.loads((dataset / "annotations" / "run_0001.json").read_text())
    assert [a["frame_index"] for a in annotations] == [0, 1, 2]
    assert annotations[0]["normal"] == "frame_000000.png"

    labels = dataset / "yolo" / "labels" / "train"
    assert (labels / "run_0001_000001.txt").read_text() == ""
    assert (labels / "run_0001_000000.txt").read_text().startswith("0 ")
    images = dataset / "yolo" / "images" / "train"
    assert len(list(images.iterdir())) == 3

    qc = tmp_path / "qc" / "run_0001"
    report = json.loads((qc / "report.json").read_text())
    assert report["run_id"] == "run_0001"
    assert len(list((qc / "debug").iterdir())) == 3


def test_unregistered_run_refused(tmp_path):
    config = registered_config(tmp_path, {0: None})
    with pytest.raises(ManifestError, match="incomplete or not a run"):
        convert_run(run_id="run_0002", config=config, dataset_version="v001")


def test_manifest_frame_count_mismatch_refused(tmp_path):
    config = registered_config(tmp_path, {0: (10, 30, 5, 15), 1: None, 2: None})
    # corrupt the run: remove one frame pair behind the manifest's back
    for side in ("normal", "mask"):
        (tmp_path / "raw" / "run_0001" / side / "frame_000002.png").unlink()
    with pytest.raises(ConvertError, match="2 frame pairs.*says 3.*corrupt"):
        convert_run(run_id="run_0001", config=config, dataset_version="v001")

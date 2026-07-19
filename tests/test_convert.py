import json

import cv2
import numpy as np

from dronesynth.config import load_convert_config
from dronesynth.datagen.convert import convert_run

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
        directory = root / side / "CameraComponent" / "ColorImage"
        directory.mkdir(parents=True)
    for index, blob in frames.items():
        normal = np.full((48, 64, 3), 90, dtype=np.uint8)
        mask = np.zeros((48, 64, 4), dtype=np.uint8)  # BGRA like EasySynth
        mask[:, :, 3] = 255
        if blob is not None:
            y0, y1, x0, x1 = blob
            mask[y0:y1, x0:x1, :3] = 255
        for side, image in (("normal", normal), ("mask", mask)):
            path = root / side / "CameraComponent" / "ColorImage" / f"seq.{index:04d}.png"
            cv2.imwrite(str(path), image)


def test_convert_run_end_to_end(tmp_path):
    write_capture(tmp_path, {0: (10, 30, 5, 15), 1: None, 2: (20, 40, 30, 50)})
    config_path = tmp_path / "convert.yaml"
    config_path.write_text(CONFIG.format(root=tmp_path))
    config = load_convert_config(config_path)

    result = convert_run(
        run_id="run_0001",
        normal_root=tmp_path / "normal",
        mask_root=tmp_path / "mask",
        config=config,
        dataset_version="v001",
    )

    assert result.report.frames == 3
    assert result.report.empty_frames == 1
    assert result.report.total_boxes == 2

    dataset = tmp_path / "datasets" / "v001"
    annotations = json.loads((dataset / "annotations" / "run_0001.json").read_text())
    assert [a["frame_index"] for a in annotations] == [0, 1, 2]

    labels = dataset / "yolo" / "labels" / "train"
    assert (labels / "run_0001_000001.txt").read_text() == ""
    assert (labels / "run_0001_000000.txt").read_text().startswith("0 ")
    images = dataset / "yolo" / "images" / "train"
    assert len(list(images.iterdir())) == 3

    qc = tmp_path / "qc" / "run_0001"
    report = json.loads((qc / "report.json").read_text())
    assert report["run_id"] == "run_0001"
    assert len(list((qc / "debug").iterdir())) == 3

    # debug render for frame 0 must differ from the plain normal render
    debug = cv2.imread(str(qc / "debug" / "run_0001_000000.png"))
    assert (debug != 90).any()

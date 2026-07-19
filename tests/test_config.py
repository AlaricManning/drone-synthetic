from pathlib import Path

import pytest

from dronesynth.config import ConfigError, load_convert_config

REPO_CONFIG = Path(__file__).parent.parent / "configs" / "convert.yaml"

VALID = """\
class_map:
  0: drone
mask:
  threshold: 12
  min_box_area: 16
split:
  mode: by_run
  val_runs: []
storage:
  raw_root: data/raw
  dataset_root: data/datasets
  qc_root: data/qc
"""


def write_config(tmp_path, text):
    path = tmp_path / "convert.yaml"
    path.write_text(text)
    return path


def test_loads_repo_config():
    config = load_convert_config(REPO_CONFIG)
    assert config.class_map == {0: "drone"}
    assert config.mask.threshold == 12
    assert config.split.mode == "by_run"


def test_loads_valid_config(tmp_path):
    config = load_convert_config(write_config(tmp_path, VALID))
    assert config.mask.min_box_area == 16
    assert config.split.val_runs == ()
    assert config.storage.raw_root == "data/raw"


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_convert_config(Path("does/not/exist.yaml"))


def test_missing_section(tmp_path):
    text = VALID.replace("mask:\n  threshold: 12\n  min_box_area: 16\n", "")
    with pytest.raises(ConfigError, match="'mask'"):
        load_convert_config(write_config(tmp_path, text))


def test_threshold_out_of_range(tmp_path):
    text = VALID.replace("threshold: 12", "threshold: 255")
    with pytest.raises(ConfigError, match="threshold"):
        load_convert_config(write_config(tmp_path, text))


def test_rejects_frame_level_split(tmp_path):
    with pytest.raises(ConfigError, match="by_run"):
        load_convert_config(write_config(tmp_path, VALID.replace("mode: by_run", "mode: random")))


def test_rejects_negative_class_id(tmp_path):
    with pytest.raises(ConfigError, match="class_map"):
        load_convert_config(write_config(tmp_path, VALID.replace("0: drone", "-1: drone")))

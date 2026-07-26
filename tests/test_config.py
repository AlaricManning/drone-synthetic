from pathlib import Path

import pytest

from dronesynth.config import ConfigError, load_build_config, load_convert_config

REPO_CONFIG = Path(__file__).parent.parent / "configs" / "convert.yaml"

VALID = """\
class_map:
  0: drone
mask:
  threshold: 12
  min_box_area: 16
storage:
  raw_root: data/raw
  dataset_root: data/datasets
  qc_root: data/qc
"""

VALID_BUILD = """\
runs:
  - run_0001
split:
  mode: by_run
  val_runs: []
storage:
  raw_root: data/raw
  dataset_root: data/datasets
"""


def write_config(tmp_path, text):
    path = tmp_path / "convert.yaml"
    path.write_text(text)
    return path


def test_loads_repo_config():
    config = load_convert_config(REPO_CONFIG)
    assert config.class_map == {0: "drone"}
    # High enough to clear the temporal-AA ghosts in runs captured before
    # 2026-07-26, which read 13-31 and would be grouped into the drone's box.
    assert config.mask.threshold == 32


def test_loads_valid_config(tmp_path):
    config = load_convert_config(write_config(tmp_path, VALID))
    assert config.mask.min_box_area == 16
    assert config.storage.raw_root == "data/raw"


def test_a_convert_config_needs_no_split(tmp_path):
    """Splits belong to dataset versions, so a conversion never declares one."""
    config = load_convert_config(write_config(tmp_path, VALID))
    assert not hasattr(config, "split")


def test_a_leftover_split_section_is_ignored(tmp_path):
    """An unedited config from before splits moved out must still convert."""
    stale = VALID + "split:\n  mode: by_run\n  val_runs: [run_0009]\n"
    config = load_convert_config(write_config(tmp_path, stale))
    assert config.mask.threshold == 12


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


def test_build_config_rejects_frame_level_split(tmp_path):
    """Frames from one camera path are near-duplicates, so a frame split leaks."""
    path = tmp_path / "build.yaml"
    path.write_text(VALID_BUILD.replace("mode: by_run", "mode: random"))
    with pytest.raises(ConfigError, match="by_run"):
        load_build_config(path)


def test_rejects_negative_class_id(tmp_path):
    with pytest.raises(ConfigError, match="class_map"):
        load_convert_config(write_config(tmp_path, VALID.replace("0: drone", "-1: drone")))

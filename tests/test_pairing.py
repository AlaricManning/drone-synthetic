import pytest

from dronesynth.datagen.pairing import PairingError, discover_frames, pair_frames


def make_capture(root, side, indices, nested=True, stem="testSequence"):
    """Lay out fake EasySynth output: <side>/CameraComponent/ColorImage/<stem>.NNNN.png"""
    directory = root / side
    if nested:
        directory = directory / "CameraComponent" / "ColorImage"
    directory.mkdir(parents=True, exist_ok=True)
    for i in indices:
        (directory / f"{stem}.{i:04d}.png").touch()
    return root / side


def test_discovers_nested_frames(tmp_path):
    root = make_capture(tmp_path, "normal", range(3))
    frames = discover_frames(root)
    assert sorted(frames) == [0, 1, 2]
    assert frames[2].name == "testSequence.0002.png"


def test_discover_missing_dir(tmp_path):
    with pytest.raises(PairingError, match="not found"):
        discover_frames(tmp_path / "nope")


def test_discover_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PairingError, match="no PNG frames"):
        discover_frames(tmp_path / "empty")


def test_discover_rejects_unindexed_filename(tmp_path):
    root = make_capture(tmp_path, "normal", [0])
    (root / "CameraComponent" / "ColorImage" / "thumbnail.png").touch()
    with pytest.raises(PairingError, match="thumbnail.png"):
        discover_frames(root)


def test_discover_rejects_duplicate_index(tmp_path):
    root = make_capture(tmp_path, "normal", [7])
    make_capture(tmp_path, "normal", [7], stem="otherSequence")
    with pytest.raises(PairingError, match="duplicate frame index 7"):
        discover_frames(root)


def test_pairs_matching_captures(tmp_path):
    normal = make_capture(tmp_path, "normal", range(5))
    mask = make_capture(tmp_path, "mask", range(5))
    pairs = pair_frames(normal, mask)
    assert [p.index for p in pairs] == [0, 1, 2, 3, 4]
    assert all(p.normal != p.mask for p in pairs)


def test_rejects_incomplete_capture(tmp_path):
    normal = make_capture(tmp_path, "normal", range(5))
    mask = make_capture(tmp_path, "mask", [0, 1, 3, 4])  # frame 2 missing
    with pytest.raises(PairingError, match=r"incomplete.*1 frame\(s\) missing a mask.*\[2\]"):
        pair_frames(normal, mask)


def test_rejects_extra_mask_frame(tmp_path):
    normal = make_capture(tmp_path, "normal", range(3))
    mask = make_capture(tmp_path, "mask", range(4))
    with pytest.raises(PairingError, match=r"missing a normal render.*\[3\]"):
        pair_frames(normal, mask)

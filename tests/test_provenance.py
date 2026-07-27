import json
import subprocess

from dronesynth.datagen.contrast import metric_config
from dronesynth.provenance import (
    ENV_COMMIT,
    ENV_DIRTY,
    PROVENANCE_SUFFIX,
    REPO_NAME,
    RunProvenance,
    converter_stamp,
    read_provenance,
    run_provenance,
    write_provenance,
)


def stamp(repo_root=None):
    """converter_stamp is process-cached, which tests must not inherit."""
    converter_stamp.cache_clear()
    try:
        return converter_stamp(repo_root) if repo_root else converter_stamp()
    finally:
        converter_stamp.cache_clear()


def test_stamp_reads_the_baked_in_commit(monkeypatch, tmp_path):
    """Production path: running from an image, which has no .git."""
    monkeypatch.setenv(ENV_COMMIT, "abc1234567")
    monkeypatch.setenv(ENV_DIRTY, "false")

    assert stamp(tmp_path) == {"repo": REPO_NAME, "commit": "abc1234567", "dirty": False}


def test_baked_in_stamp_records_a_dirty_build(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_COMMIT, "abc1234567")
    monkeypatch.setenv(ENV_DIRTY, "true")

    assert stamp(tmp_path)["dirty"] is True


def test_unbuilt_args_are_not_mistaken_for_a_commit(monkeypatch, tmp_path):
    """A bare `docker build` leaves the args at "unknown"; that is not a commit."""
    monkeypatch.setenv(ENV_COMMIT, "unknown")
    monkeypatch.setenv(ENV_DIRTY, "unknown")

    # Falls through to git, which finds nothing in a bare tmp dir.
    assert stamp(tmp_path) == {"repo": REPO_NAME, "commit": None, "dirty": None}


def test_stamp_falls_back_to_git_in_a_checkout(monkeypatch):
    """Development path: no baked-in stamp, but a real checkout to ask."""
    monkeypatch.delenv(ENV_COMMIT, raising=False)
    monkeypatch.delenv(ENV_DIRTY, raising=False)

    result = stamp()  # module default: this repo
    assert result["repo"] == REPO_NAME
    assert result["commit"] is not None
    assert len(result["commit"]) == 10
    assert isinstance(result["dirty"], bool)


def make_repo(tmp_path):
    """A throwaway checkout, to exercise the dirty flag without touching this one."""
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", *args], cwd=tmp_path, capture_output=True, check=True
    )
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "README.md").write_text("docs\n")
    run("add", "-A")
    run("commit", "-qm", "initial")
    return tmp_path


def test_dirty_is_scoped_to_the_installed_code(monkeypatch, tmp_path):
    """A README edit cannot make the converter disagree with the commit it names."""
    monkeypatch.delenv(ENV_COMMIT, raising=False)
    monkeypatch.delenv(ENV_DIRTY, raising=False)
    repo = make_repo(tmp_path)

    assert stamp(repo)["dirty"] is False

    (repo / "README.md").write_text("docs, revised\n")
    assert stamp(repo)["dirty"] is False

    (repo / "src" / "mod.py").write_text("x = 2\n")
    assert stamp(repo)["dirty"] is True


def test_dirty_counts_untracked_code(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_COMMIT, raising=False)
    monkeypatch.delenv(ENV_DIRTY, raising=False)
    repo = make_repo(tmp_path)

    (repo / "src" / "stray.py").write_text("y = 1\n")
    assert stamp(repo)["dirty"] is True


# The line-ending half of this is not unit-tested on purpose. It reproduces only
# when the reader's git and the checkout's git disagree about normalization --
# WSL reading a Windows worktree -- and a temp repo created by whichever git is
# running the tests cannot manufacture that disagreement. Verified directly
# instead: `git status --porcelain` reported all 56 files modified there, and
# --ignore-cr-at-eol reported clean.


def test_stamp_records_absence_rather_than_guessing(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_COMMIT, raising=False)
    monkeypatch.delenv(ENV_DIRTY, raising=False)

    assert stamp(tmp_path) == {"repo": REPO_NAME, "commit": None, "dirty": None}


def test_stamp_is_cached_across_a_batch(monkeypatch):
    """Editing the tree mid-batch must not change what later runs claim."""
    monkeypatch.setenv(ENV_COMMIT, "aaaaaaaaaa")
    converter_stamp.cache_clear()
    try:
        first = converter_stamp()
        monkeypatch.setenv(ENV_COMMIT, "bbbbbbbbbb")
        assert converter_stamp() == first
    finally:
        converter_stamp.cache_clear()


def test_run_provenance_records_the_mask_config():
    record = run_provenance("run_0001", threshold=32, min_box_area=4, class_map={0: "drone"})

    assert record.conversion == {
        "threshold": 32,
        "min_box_area": 4,
        "class_map": {"0": "drone"},
        "contrast": metric_config(),
    }
    assert record.run_id == "run_0001"
    assert record.converted_at.endswith("+00:00")


def test_class_map_keys_survive_json_comparably(tmp_path):
    """The build compares conversion blocks; int and str keys must not differ."""
    fresh = run_provenance("run_0001", threshold=32, min_box_area=4, class_map={0: "drone"})
    path = tmp_path / f"run_0001{PROVENANCE_SUFFIX}"
    write_provenance(fresh, path)

    assert read_provenance(path).conversion == fresh.conversion


def test_provenance_round_trip(tmp_path):
    original = RunProvenance(
        run_id="run_0001",
        converted_at="2026-07-26T12:00:00+00:00",
        converter={"repo": REPO_NAME, "commit": "abc1234567", "dirty": False},
        conversion={"threshold": 32, "min_box_area": 4, "class_map": {"0": "drone"}},
    )
    path = tmp_path / "annotations" / f"run_0001{PROVENANCE_SUFFIX}"
    write_provenance(original, path)

    assert read_provenance(path) == original
    # Readable without this module, since an audit may happen from anywhere.
    assert json.loads(path.read_text())["converter"]["commit"] == "abc1234567"


def test_provenance_tolerates_missing_blocks():
    """Read back what an older or partial writer produced without crashing."""
    assert RunProvenance.from_dict({"run_id": "r", "converted_at": "t"}) == RunProvenance(
        run_id="r", converted_at="t", converter={}, conversion={}
    )

import pytest

from dronesynth.datagen.split import SplitError, split_runs


def test_all_train_when_no_val_runs():
    assert split_runs(["run_0001"], []) == {"run_0001": "train"}


def test_val_runs_assigned():
    assignments = split_runs(["run_0001", "run_0002", "run_0003"], ["run_0002"])
    assert assignments == {"run_0001": "train", "run_0002": "val", "run_0003": "train"}


def test_unknown_val_run_rejected():
    with pytest.raises(SplitError, match="run_0009"):
        split_runs(["run_0001"], ["run_0009"])

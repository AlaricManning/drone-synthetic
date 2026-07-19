"""Run-level train/val assignment.

Frames within one run are near-duplicates (consecutive frames of one camera
path), so splitting is only ever done at run granularity: whole runs are held
out for validation. Frame-level splitting would leak train data into val.
"""

from __future__ import annotations

from collections.abc import Iterable


class SplitError(ValueError):
    """Raised when the requested split doesn't match the input runs."""


def split_runs(run_ids: Iterable[str], val_runs: Iterable[str]) -> dict[str, str]:
    """Assign each run to 'train' or 'val' per the configured val_runs list."""
    run_ids = list(run_ids)
    val_set = set(val_runs)
    unknown = sorted(val_set - set(run_ids))
    if unknown:
        raise SplitError(f"val_runs not among the input runs: {unknown}")
    return {run_id: "val" if run_id in val_set else "train" for run_id in run_ids}

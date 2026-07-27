"""Which build of the converter produced a set of labels, and under what config.

The render side already records this: every run manifest carries a ``generator``
stamp naming the commit that made the pixels. One stage downstream the same gap
was wide open. A dataset recorded what was asked for — the runs, the class map —
and nothing about the code that turned masks into boxes.

That is not hypothetical. On 2026-07-26 the mask threshold moved from 12 to 32
and instance grouping landed, which together changed the content of every label
this pipeline emits. Labels from either side of that afternoon are materially
different and were indistinguishable in the artifacts. Six test runs converted
under the old threshold still sit in the same bucket as the corpus.

Two things are recorded, because either alone is insufficient. The commit says
what code ran; the mask config says what that code was told to do. A build
assembling runs into a dataset compares these across its inputs and refuses to
mix, since labels made under different thresholds are not comparable.

Where the commit comes from
---------------------------
The render side can simply ask git, because it runs from a checkout. The
converter cannot: in production it runs from a wheel installed into a container
whose image contains ``src``, ``configs`` and no ``.git`` at all. A purely
git-based stamp would therefore report "unknown" for every real conversion and
succeed only in development, which is precisely backwards.

So the commit is baked into the image at build time and read from the
environment here, with git as the fallback for development. ``scripts/build_image.sh``
computes the build args so the documented path cannot forget them; a bare
``docker build`` still works and yields an honest "unknown" rather than a
plausible-looking wrong answer.

Absence is recorded rather than guessed at. A tree that is not a git checkout,
or an image built without the args, yields ``None`` instead of a placeholder, so
downstream can tell "converted at this commit" from "we do not know". Treating
those alike is the mistake this exists to prevent.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dronesynth.datagen.contrast import metric_config

logger = logging.getLogger(__name__)

REPO_NAME = "drone-synthetic"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Long enough to stay unambiguous as history grows, short enough to read.
COMMIT_LENGTH = 10

# Baked into the conversion image; see docker/Dockerfile.
ENV_COMMIT = "DRONESYNTH_GIT_COMMIT"
ENV_DIRTY = "DRONESYNTH_GIT_DIRTY"

# What the Dockerfile defaults to when the build args are not supplied.
UNKNOWN = "unknown"

# Written beside the annotations it describes: <run_id>.provenance.json next to
# <run_id>.json. A sidecar rather than a field inside the annotations file
# because write_annotations emits a bare JSON list, and wrapping it would break
# read_annotations and every file already in the bucket.
PROVENANCE_SUFFIX = ".provenance.json"

_TRUTHY = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if result.returncode != 0:
        logger.debug("git %s exited %s", " ".join(args), result.returncode)
        return None
    return result.stdout.strip()


def _from_env() -> dict[str, Any] | None:
    """The stamp baked into the image, if this is running from one."""
    commit = os.environ.get(ENV_COMMIT, "").strip()
    if not commit or commit == UNKNOWN:
        return None
    dirty = os.environ.get(ENV_DIRTY, "").strip().lower()
    return {
        "repo": REPO_NAME,
        "commit": commit,
        "dirty": _TRUTHY.get(dirty),
    }


# What the package installs, so what a difference from HEAD could actually
# change about a conversion. An edited README cannot.
CODE_PATHS = ["src", "pyproject.toml"]


def _is_dirty(repo_root: Path) -> bool | None:
    """Whether the installed code differs from HEAD.

    Deliberately not `git status --porcelain`, which on a Windows checkout read
    from WSL calls every file modified: the worktree holds CRLF and the index
    holds LF. That would peg the flag at true and discredit stamps that are in
    fact exact, so compare ignoring the carriage return.
    """
    if _git(["rev-parse", "--is-inside-work-tree"], repo_root) != "true":
        return None
    modified = _git(["diff", "--quiet", "--ignore-cr-at-eol", "HEAD", "--", *CODE_PATHS], repo_root)
    if modified is None:  # non-zero exit: something under CODE_PATHS differs
        return True
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS], repo_root)
    return bool(untracked)


def _from_git(repo_root: Path) -> dict[str, Any]:
    commit = _git(["rev-parse", f"--short={COMMIT_LENGTH}", "HEAD"], repo_root)
    return {
        "repo": REPO_NAME,
        "commit": commit,
        # Uncommitted changes mean the commit names roughly what ran, not
        # exactly. Worth knowing before labels are used to argue anything.
        "dirty": None if commit is None else _is_dirty(repo_root),
    }


@lru_cache(maxsize=1)
def converter_stamp(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The converter's identity: repo, commit, and whether the tree was dirty.

    Cached for the life of the process, which is also the more correct
    behaviour: editing the tree partway through a batch should not make later
    outputs claim a commit whose changes never reached the converter.
    """
    return _from_env() or _from_git(repo_root)


@dataclass(frozen=True)
class RunProvenance:
    """What produced one run's labels."""

    run_id: str
    converted_at: str
    converter: dict[str, Any] = field(default_factory=dict)
    conversion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "converted_at": self.converted_at,
            "converter": self.converter,
            "conversion": self.conversion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunProvenance:
        return cls(
            run_id=data["run_id"],
            converted_at=data["converted_at"],
            converter=data.get("converter", {}),
            conversion=data.get("conversion", {}),
        )


def run_provenance(
    run_id: str, *, threshold: int, min_box_area: int, class_map: dict
) -> RunProvenance:
    """Assemble the record for a conversion happening now."""
    return RunProvenance(
        run_id=run_id,
        converted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        converter=converter_stamp(),
        conversion={
            "threshold": threshold,
            "min_box_area": min_box_area,
            # Not a setting anyone passes, but a definition: contrast numbers
            # are only comparable between runs measured the same way, and the
            # metric is new enough to expect revision. Recording it here is what
            # makes the build's "converted alike" check cover it too.
            "contrast": metric_config(),
            # Keys stringified up front so a config compared straight from a
            # ConvertConfig matches one read back out of JSON. Without this the
            # build's "were these converted alike" check would see {0: 'drone'}
            # and {'0': 'drone'} as different.
            "class_map": {str(k): v for k, v in class_map.items()},
        },
    )


def write_provenance(provenance: RunProvenance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance.to_dict(), indent=2))


def read_provenance(path: Path) -> RunProvenance:
    return RunProvenance.from_dict(json.loads(path.read_text()))

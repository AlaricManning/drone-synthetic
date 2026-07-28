"""Generate a build config with a stratified, blind validation split.

    python scripts/make_build_config.py v004 --seed-prefix 3

Reads every candidate run's manifest from the raw prefix, groups by weather
preset and clip type, and holds out a fixed number of each.

Two properties, both learned from v003.

*Stratified.* A corpus spanning ten atmospheres validated on a random draw can
easily hold out four clear-sky runs and no rain. Taking a fixed number from
each preset and each clip type makes the hold-out cover what the training set
covers, and makes per-condition numbers readable rather than accidental.

*Blind.* v003's hold-out was drawn by replacing runs that carried low-contrast
frames, which selected for high contrast and left validation measurably easier
than the data it validated -- its median contrast was -47.1 against the
corpus's -33.2. So this draws on run identity alone and looks at nothing about
the frames. A split that knows what it is selecting for will select for it.

The draw is seeded, so the config regenerates identically and a reviewer can
check the split rather than take it on trust.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DRAW_SEED = 20260728


def load(bucket: str, run_id: str) -> tuple[str, dict]:
    import boto3

    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=f"raw/{run_id}/manifest.json")["Body"].read()
    return run_id, json.loads(body)


# A recession multiplies its distance; a crossing holds it. The ratio is the
# discriminator rather than the difference, because both endpoints are jittered
# independently, so a crossing authored at a constant 6000 cm arrives with ends
# thousands of centimetres apart and an absolute threshold misreads it.
#
# Measured over this corpus, where the schedule supplies ground truth: 100
# recessions span ratios 18.2 to 76.2 and 50 crossings span 0.46 to 2.23. There
# is an order of magnitude of empty space between them.
RECESSION_RATIO = 5.0


def clip_kind(rand: dict) -> str:
    start, end = rand.get("distance_start_cm"), rand.get("distance_end_cm")
    if not start or end is None:
        return "unknown"
    return "recession" if float(end) / float(start) >= RECESSION_RATIO else "crossing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--bucket", default="drone-synthetic-am")
    ap.add_argument("--seed-prefix", default="3",
                    help="first digit of the seeds belonging to this corpus")
    ap.add_argument("--val-recession", type=int, default=2)
    ap.add_argument("--val-crossing", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import boto3

    s3 = boto3.client("s3")
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=args.bucket, Prefix="raw/", Delimiter="/"
    )
    run_ids = [
        p["Prefix"].split("/")[1]
        for page in pages
        for p in page.get("CommonPrefixes", [])
    ]
    corpus = sorted(r for r in run_ids if r.rsplit("_", 1)[-1].startswith(args.seed_prefix)
                    and len(r.rsplit("_", 1)[-1]) == 5)
    print(f"{len(corpus)} runs match seed prefix {args.seed_prefix}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = dict(pool.map(lambda r: load(args.bucket, r), corpus))

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for run_id, manifest in loaded.items():
        rand = manifest.get("randomization") or {}
        preset = rand.get("preset") or "unknown"
        groups[(preset, clip_kind(rand))].append(run_id)

    rng = random.Random(DRAW_SEED)
    val: list[tuple[str, str, str]] = []
    for (preset, kind), members in sorted(groups.items()):
        want = args.val_recession if kind == "recession" else args.val_crossing
        for run_id in rng.sample(sorted(members), min(want, len(members))):
            val.append((run_id, preset, kind))
    val.sort()
    val_ids = {v[0] for v in val}

    print(f"{len(corpus) - len(val_ids)} train, {len(val_ids)} val\n")
    for (preset, kind), members in sorted(groups.items()):
        held = sum(1 for v in val if v[1] == preset and v[2] == kind)
        print(f"  {preset:18} {kind:10} {len(members):3d} runs, {held} held out")

    lines = _render(args.version, corpus, val, loaded, groups, args)
    out = args.out or Path(f"configs/build.{args.version}.s3.yaml")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _render(version, corpus, val, loaded, groups, args) -> list[str]:
    by_preset: dict[str, list[str]] = defaultdict(list)
    for run_id in corpus:
        rand = loaded[run_id].get("randomization") or {}
        by_preset[rand.get("preset") or "unknown"].append(run_id)

    lines = [
        f"# Dataset version {version}: the weather corpus.",
        "#",
        f"# {len(corpus)} runs across {len(by_preset)} Ultra Dynamic Weather presets,"
        " dealt evenly rather than",
        f"# sampled -- {len(corpus) // len(by_preset)} runs each, ten recession and five"
        " crossing. Every earlier version",
        "# was a single atmosphere the manifests described inaccurately; this is",
        "# the first where the recorded weather is what rendered, verified by",
        "# reading each dimension back from the actor.",
        "#",
        "# One config per version, tracked in git, because which runs a dataset",
        "# trains on is the decision most worth being able to look up later.",
        "#",
        "# No class_map and no mask settings: those belong to the conversion that",
        "# produced the annotations, and the build reads them from each run's",
        "# provenance rather than restating them where they could contradict it.",
        "",
        "runs:",
    ]
    for preset in sorted(by_preset):
        lines.append(f"  # {preset}")
        lines += [f"  - {run_id}" for run_id in sorted(by_preset[preset])]
        lines.append("")

    lines += [
        "split:",
        "  # Whole runs, never frames: consecutive frames of one camera path are",
        "  # near-duplicates, so a frame-level split leaks train into val.",
        "  mode: by_run",
        "",
        f"  # {len(val)} runs, {args.val_recession} recession and {args.val_crossing} crossing"
        " from every preset.",
        "  #",
        "  # Stratified because a random draw over ten atmospheres can hold out four",
        "  # clear-sky runs and no rain, which makes any per-condition number an",
        "  # accident. Taking a fixed count from each preset makes the hold-out cover",
        "  # what the training set covers.",
        "  #",
        "  # Blind, which is the correction to v003. That split replaced runs carrying",
        "  # low-contrast frames, and selecting against low contrast selects for high",
        "  # contrast: its median sat at -47.1 grey levels against the corpus's -33.2,",
        "  # so it was measurably easier than the data it validated. This draw sees",
        "  # only run identity and a seeded RNG. Nothing about the frames enters it.",
        "  #",
        "  # Still one drone model against one sky, and run-level splitting means the",
        "  # effective size is 30 scenes rather than 1800 frames. It measures whether",
        "  # a detector holds up across weather; it does not establish that one",
        "  # generalizes to real footage.",
        "  val_runs:",
    ]
    width = max(len(v[0]) for v in val)
    for run_id, preset, kind in val:
        lines.append(f"    - {run_id:{width}}   # {preset}, {kind}")

    lines += [
        "",
        "storage:",
        "  raw_root: s3://drone-synthetic-am/raw",
        "  dataset_root: s3://drone-synthetic-am/datasets",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

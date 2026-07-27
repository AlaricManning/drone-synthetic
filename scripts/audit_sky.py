"""Did any run in a dataset render a sky that is not its own?

drone-synth-render saves the level before handing off to Movie Render Queue,
which renders in a separate process and reads that file. Unreal reports a failed
save by returning False rather than raising, so for most of the project's
history a locked .umap made runs render whatever the last successful save left
behind -- while their manifests described the config they asked for. The
renderer refuses on a failed save now, but runs made before that are suspect.

Cloud coverage and fog never reached those frames (they were routed to the sky
actor, which the weather actor overwrites), so with the drone out of the way the
sky is a function of time of day and nothing else. A run whose sky does not
match its requested hour rendered someone else's.

    python scripts/audit_sky.py configs/build.v003.s3.yaml

Downloads one mid-clip frame per run. Frames are large and only the sky is
needed, so this samples rather than syncing the corpus.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
import yaml
from botocore.config import Config
from PIL import Image

FRAME_INDEX = 30  # mid-clip, past any auto-exposure settling
SKY_ROWS = 300  # top of frame: sky only, never the drone or the horizon
NEIGHBOURS = 5  # runs either side in time to compare against
SUSPICIOUS = 15.0  # blue-red units; honest runs sit within a couple
FLAT_ENOUGH = 5.0  # neighbour disagreement below which the test is trustworthy
SAME_HOUR = 0.15  # hours apart still counting as the same request
AGREES = 5.0  # blue-red units within which two runs corroborate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path, help="build config listing the runs")
parser.add_argument("--bucket", default="drone-synthetic-am")
parser.add_argument("--cache", type=Path, default=Path("audit"))
args = parser.parse_args()

cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
runs = cfg["runs"]
val = set(cfg.get("split", {}).get("val_runs", []))
s3 = boto3.client("s3", config=Config(max_pool_connections=32))


def fetch(run: str) -> None:
    d = args.cache / run
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "manifest.json").exists():
        s3.download_file(args.bucket, f"raw/{run}/manifest.json", str(d / "manifest.json"))
    if not (d / "frame.png").exists():
        listing = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"raw/{run}/normal/")
        keys = sorted(o["Key"] for o in listing.get("Contents", []))
        if keys:
            s3.download_file(
                args.bucket, keys[min(FRAME_INDEX, len(keys) - 1)], str(d / "frame.png")
            )


print(f"sampling one frame from each of {len(runs)} runs")
with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(fetch, runs))

rows = []
for run in runs:
    d = args.cache / run
    if not (d / "frame.png").exists():
        print(f"  no frames for {run}")
        continue
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    # Ingest schema: the renderer's nested uds block arrives flattened here.
    tod = (m.get("randomization") or {}).get("time_of_day")
    if tod is None:
        continue
    sky = np.asarray(Image.open(d / "frame.png").convert("RGB")).astype(np.float64)[:SKY_ROWS]
    rows.append(
        {
            "run": run,
            "seed": m.get("seed"),
            "tod": tod,
            "br": float((sky[:, :, 2] - sky[:, :, 0]).mean()),
        }
    )

rows.sort(key=lambda r: r["tod"])
brs = np.array([r["br"] for r in rows])
print(f"\n{len(rows)} runs, time of day {rows[0]['tod']:.2f} to {rows[-1]['tod']:.2f}\n")
print(f"{'run':32} {'seed':>5} {'t.o.d':>6} {'blue-red':>9} {'expected':>9} {'off by':>8}")
print("-" * 82)

flagged, unclear = [], []
for i, r in enumerate(rows):
    lo, hi = max(0, i - NEIGHBOURS), min(len(rows), i + NEIGHBOURS + 1)
    others = np.array([brs[j] for j in range(lo, hi) if j != i])
    expected = float(np.median(others))
    off = r["br"] - expected

    # How much do the neighbours disagree among themselves? Through the middle
    # of the day the sky barely moves and they agree within a unit or two, so a
    # large deviation is real. Towards evening it falls off a cliff -- +27 to
    # -32 inside an hour -- and neighbours differ by tens for honest reasons.
    testable = float(np.median(np.abs(others - expected))) < FLAT_ENOUGH

    # A second run that asked for the same hour and got the same sky settles it
    # whatever the neighbourhood says: two independent renders agreeing is the
    # thing the neighbour median stands in for.
    twin = next(
        (
            o
            for o in rows
            if o is not r
            and abs(o["tod"] - r["tod"]) < SAME_HOUR
            and abs(o["br"] - r["br"]) < AGREES
        ),
        None,
    )

    mark = ""
    if abs(off) > SUSPICIOUS:
        if twin is not None:
            mark = f"  (corroborated by {twin['seed']} at the same hour)"
        elif testable:
            mark = "  <-- WRONG SKY"
            flagged.append(r)
        else:
            mark = "  (evening cliff, cannot tell)"
            unclear.append(r)
    print(
        f"{r['run']:32} {r['seed']:5} {r['tod']:6.2f} {r['br']:9.2f} "
        f"{expected:9.2f} {off:+8.2f}{mark}"
    )

print(f"\n{len(flagged)} run(s) definitely rendered the wrong sky")
for r in flagged:
    twin = min((o for o in rows if o is not r), key=lambda o: abs(o["br"] - r["br"]))
    print(
        f"  {r['run']}  asked for {r['tod']:.2f}, looks like {twin['seed']} "
        f"at {twin['tod']:.2f}  [{'VAL' if r['run'] in val else 'train'}]"
    )
if unclear:
    print(
        f"\n{len(unclear)} on the evening falloff, where the test cannot decide: "
        + ", ".join(str(r["seed"]) for r in unclear)
    )
if flagged:
    hit = sum(1 for r in flagged if r["run"] in val)
    print(f"\n{hit} of {len(val)} validation runs affected")
    print(
        "\nLabels are unaffected: boxes come from the mask pass, which depends on\n"
        "the drone's geometry and position, and those arrive through the per-run\n"
        "sequence asset that saves separately. What is wrong is the metadata."
    )

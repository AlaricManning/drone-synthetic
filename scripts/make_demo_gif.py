"""Record one real run walking the whole pipeline, for the README.

A diagram of a pipeline is cheap; a recording of one working is not. So this
takes a run that was actually rendered and uploaded, waits for the trigger chain
to convert it, and composes the result -- job id, timings, box coordinates and
all -- into an animation. Nothing in the output is illustrative.

    # from drone-synth-render, with DRS_ENV=prod
    python -m drone_synth_render.cli render --config configs/demo_run.yaml \
        --pass both --upload

    # then here, while the conversion is still queueing
    python scripts/make_demo_gif.py run_20260727_150314_7001 \
        --frames ../drone-synth-render/runs/run_20260727_150314_7001

The drone is a few dozen pixels across in a 1920x1080 frame and disappears
entirely at README scale, so every stage carries a zoom inset anchored on it.
A panel without one would be a photograph of empty sky.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from PIL import Image, ImageDraw, ImageFont

W, H = 880, 384
BAR_H, CAP_H = 44, 38

# A GIF cannot be paused, so the three text stages are drawn complete and held
# rather than typed in line by line. Revealing them a line at a time reads as
# activity and gives the eye something to chase, which is exactly what makes a
# panel hard to follow when you cannot stop it. The MP4 has real controls, and
# is written from these same frames.
HOLD = 3200
CLIP_MS = 90

INK = (232, 236, 243)
DIM = (139, 148, 158)
BG = (13, 17, 23)
PANEL = (22, 27, 34)
EDGE = (48, 54, 61)
LIVE = (88, 166, 255)
DONE = (63, 185, 80)
# The same green the QC debug render draws, so a box here and a box in the
# stills below it in the README read as the same thing rather than two.
BOX = (0, 255, 0)

STAGES = ["render", "upload", "convert", "dataset"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", help="run id, already rendered and uploaded")
parser.add_argument("--frames", type=Path, required=True, help="local run directory")
parser.add_argument("--bucket", default="drone-synthetic-am")
parser.add_argument("--queue", default="dronesynth-convert")
parser.add_argument("--dataset-version", default="v003")
parser.add_argument("--out", type=Path, default=Path("assets/pipeline-demo.gif"))
args = parser.parse_args()

RUN = args.run
batch = boto3.client("batch", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")


def watch_chain() -> dict:
    """Wait for the run's conversion and report what the chain actually did."""
    job, deadline = None, time.time() + 300
    while time.time() < deadline and job is None:
        for status in (
            "SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED",
        ):
            hits = [
                j
                for j in batch.list_jobs(jobQueue=args.queue, jobStatus=status).get(
                    "jobSummaryList", []
                )
                if RUN in j.get("jobName", "")
            ]
            if hits:
                job = hits[0]
                break
        if job is None:
            print("  waiting for the trigger to fire...")
            time.sleep(10)
    if job is None:
        raise SystemExit(f"no conversion job for {RUN}: the chain did not fire")

    print(f"triggered: {job['jobName']}")
    last = None
    while True:
        d = batch.describe_jobs(jobs=[job["jobId"]])["jobs"][0]
        if d["status"] != last:
            last = d["status"]
            print(f"  {last}")
        if last in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(10)
    if last == "FAILED":
        raise SystemExit(f"conversion failed: {d.get('statusReason')}")

    key = f"datasets/auto-{RUN}/annotations/{RUN}.provenance.json"
    prov = json.loads(s3.get_object(Bucket=args.bucket, Key=key)["Body"].read())
    return {
        "job_id": job["jobId"],
        "queued_s": round(d["startedAt"] / 1000 - d["createdAt"] / 1000),
        "ran_s": round(d["stoppedAt"] / 1000 - d["startedAt"] / 1000),
        "converter": prov.get("converter", {}),
    }


chain = watch_chain()
anns = json.loads(
    s3.get_object(
        Bucket=args.bucket, Key=f"datasets/auto-{RUN}/annotations/{RUN}.json"
    )["Body"].read()
)
by_index = {a["frame_index"]: a for a in anns}

totals = json.loads(
    s3.get_object(
        Bucket=args.bucket, Key=f"datasets/{args.dataset_version}/manifest.json"
    )["Body"].read()
)["totals"]


def font(size: int, mono: bool = True) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "cour.ttf") if mono else ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


F_BAR, F_BODY, F_CAP, F_TINY = font(14, False), font(13), font(12), font(11)


def shell(stage: int, caption: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """The frame every stage shares: progress across the top, caption below."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    x = 24
    for i, name in enumerate(STAGES):
        d.ellipse([x, 18, x + 9, 27], fill=LIVE if i == stage else (DONE if i < stage else DIM))
        d.text((x + 17, 15), name, font=F_BAR, fill=INK if i <= stage else DIM)
        if i < len(STAGES) - 1:
            ax = x + 17 + int(d.textlength(name, font=F_BAR)) + 14
            tint = DONE if i < stage else EDGE
            d.line([ax, 23, ax + 26, 23], fill=tint)
            d.polygon([(ax + 26, 19), (ax + 32, 23), (ax + 26, 27)], fill=tint)
            x = ax + 46
    d.line([0, BAR_H - 1, W, BAR_H - 1], fill=EDGE)
    d.line([0, H - CAP_H, W, H - CAP_H], fill=EDGE)
    d.text((24, H - CAP_H + 12), caption, font=F_CAP, fill=DIM)
    return im, d


def panel(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    d.rectangle(box, fill=PANEL, outline=EDGE)
    d.text((box[0] + 10, box[1] + 7), title, font=F_TINY, fill=DIM)


def drone_box(idx: int) -> tuple[int, int, int, int] | None:
    boxes = by_index.get(idx, {}).get("boxes") or []
    return None if not boxes else (
        boxes[0]["x"], boxes[0]["y"], boxes[0]["w"], boxes[0]["h"]
    )


def render_panel(
    im: Image.Image,
    d: ImageDraw.ImageDraw,
    src: Image.Image,
    idx: int,
    at: tuple[int, int],
    size: tuple[int, int],
    title: str,
    draw_box: bool = False,
) -> None:
    pw, ph = size
    panel(d, (at[0], at[1], at[0] + pw, at[1] + ph + 22), title)
    iw, ih = pw - 16, ph - 8
    im.paste(src.resize((iw, ih), Image.LANCZOS), (at[0] + 8, at[1] + 22))

    box = drone_box(idx)
    if box is None:
        return
    bx, by, bw, bh = box
    sx, sy = iw / src.width, ih / src.height
    if draw_box:
        d.rectangle(
            [
                at[0] + 8 + bx * sx - 2, at[1] + 22 + by * sy - 2,
                at[0] + 8 + (bx + bw) * sx + 2, at[1] + 22 + (by + bh) * sy + 2,
            ],
            outline=BOX,
        )

    # A fixed crop window, so the magnification stays honest and constant
    # rather than growing as the target recedes.
    win, zs = 150, 96
    cx, cy = bx + bw // 2, by + bh // 2
    ox = max(0, min(src.width - win, cx - win // 2))
    oy = max(0, min(src.height - win, cy - win // 2))
    zoom = src.crop((ox, oy, ox + win, oy + win)).resize((zs, zs), Image.NEAREST)
    zx, zy = at[0] + pw - zs - 12, at[1] + ph + 22 - zs - 8
    im.paste(zoom, (zx, zy))
    d.rectangle([zx - 1, zy - 1, zx + zs, zy + zs], outline=LIVE)
    d.text((zx + 2, zy - 14), f"{win}px crop", font=F_TINY, fill=LIVE)
    if draw_box:
        z = zs / win
        d.rectangle(
            [
                zx + (bx - ox) * z - 1, zy + (by - oy) * z - 1,
                zx + (bx + bw - ox) * z + 1, zy + (by + bh - oy) * z + 1,
            ],
            outline=BOX,
        )


def frame_image(idx: int, sub: str) -> Image.Image:
    return Image.open(args.frames / sub / f"RunTemplate.{idx:04d}.png").convert("RGB")


frames: list[Image.Image] = []
delays: list[int] = []
n = len(anns)

# --- render: two passes, one camera path ------------------------------------
for idx in range(0, n, 3):
    im, d = shell(
        0, f"UE5 Movie Render Queue  ·  frame {idx:02d}/{n - 1}  ·  1920x1080  ·  90 warm-up ticks"
    )
    render_panel(im, d, frame_image(idx, "rgb"), idx, (24, 60), (400, 210), "normal")
    render_panel(im, d, frame_image(idx, "mask"), idx, (456, 60), (400, 210), "mask")
    frames.append(im)
    delays.append(CLIP_MS)
frames.append(frames[-1].copy())
delays.append(900)

# --- upload: the publish set landing, manifest last -------------------------
keys = (
    [f"raw/{RUN}/normal/RunTemplate.{i:04d}.png" for i in range(0, n, 7)]
    + [f"raw/{RUN}/mask/RunTemplate.{i:04d}.png" for i in range(0, n, 7)]
    + [f"raw/{RUN}/job.json", f"raw/{RUN}/manifest.json"]
)
im, d = shell(1, "manifest written last, so a reader never sees a half-run")
panel(d, (24, 60, 856, 292), f"s3://{args.bucket}/raw/{RUN}/")
y = 84
for k in keys[-15:]:
    final = k.endswith("manifest.json")
    d.text((40, y), ("PUT  " + k)[:96], font=F_TINY, fill=DONE if final else DIM)
    y += 14
d.text((40, y + 4), "^ the commit point", font=F_TINY, fill=DONE)
frames.append(im)
delays.append(HOLD)

# --- convert: triggered, not invoked ----------------------------------------
idx = min(12, n - 1)
ann = by_index[idx]["boxes"][0]
lines = [
    "s3:ObjectCreated  raw/.../manifest.json",
    "  -> EventBridge rule",
    "     -> Lambda: is this a manifest?",
    f"        -> Batch: convert-{RUN[:18]}...",
    "",
    f"  queued {chain['queued_s']}s   ran {chain['ran_s']}s   SUCCEEDED",
    "",
    "  mask -> connected components -> box",
    f"  {json.dumps({k: ann[k] for k in ('x', 'y', 'w', 'h')})}",
    f"  contrast {ann['contrast']}  ·  components {ann['components']}",
]
im, d = shell(
    2, f"nobody ran this  ·  the manifest landing triggered it  ·  job {chain['job_id'][:8]}"
)
render_panel(
    im, d, frame_image(idx, "rgb"), idx, (24, 60), (400, 210),
    "label derived from the mask, not drawn by hand", draw_box=True,
)
panel(d, (456, 60, 856, 292), "trigger chain")
y = 84
for ln in lines:
    tint = DONE if "SUCCEEDED" in ln else (LIVE if ln.startswith("  {") else DIM)
    d.text((472, y), ln, font=F_TINY, fill=tint)
    y += 16
frames.append(im)
delays.append(HOLD + 800)

# --- dataset: assembled, versioned, write-once ------------------------------
v = args.dataset_version
tr, va = totals["train_frames"], totals["val_frames"]
commit = chain["converter"].get("commit", "?")[:10]
tree = [
    "manifest.json",
    "  yolo/dataset.yaml",
    f"  yolo/images/train/  {tr}       yolo/images/val/  {va}",
    f"  yolo/labels/train/  {tr}       yolo/labels/val/  {va}",
    "",
    f"  {totals['runs']} converted runs gathered  ·  "
    f"{totals['frames']} frames  ·  {totals['boxes']} boxes",
    "  split by run, so no clip has frames on both sides",
    "  labels from threshold 32, min_box_area 16",
    f"  each run stamped with the converter that made it ({commit})",
]
im, d = shell(3, "dronesynth build  ·  write-once: a version that exists is never overwritten")
panel(d, (24, 60, 856, 292), f"s3://{args.bucket}/datasets/{v}/")
y = 88
for ln in tree:
    head = bool(ln) and not ln.startswith("  ")
    d.text((44, y), ln, font=F_BODY if head else F_TINY, fill=INK if head else DIM)
    y += 19
frames.append(im)
delays.append(HOLD + 800)

args.out.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(
    args.out, save_all=True, append_images=frames[1:], duration=delays, loop=0, optimize=True
)
print(f"\nwrote {args.out}  {len(frames)} frames  {args.out.stat().st_size / 1e6:.2f} MB")

# The MP4 exists for the play, pause and scrub the GIF cannot offer. H.264 via
# ffmpeg rather than OpenCV, whose builds here only offer mp4v -- which most
# browsers decline to play, making the controls moot.
mp4 = args.out.with_suffix(".mp4")
fps = 25
with TemporaryDirectory() as tmp:
    n_out = 0
    for im, ms in zip(frames, delays, strict=True):
        for _ in range(max(1, round(ms / 1000 * fps))):
            im.save(Path(tmp) / f"{n_out:05d}.png")
            n_out += 1
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(Path(tmp) / "%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "20",
        # Browsers need 4:2:0; without it Safari and GitHub's player show black.
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(mp4),
    ]
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not on PATH; skipping the mp4")
    else:
        subprocess.run(cmd, check=True)
        secs = n_out / fps
        print(f"wrote {mp4}  {secs:.1f}s  {mp4.stat().st_size / 1e6:.2f} MB")

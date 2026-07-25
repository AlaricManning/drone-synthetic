"""Compose the README demo images out of real pipeline output.

The pipeline's claim is that a mask render yields exact boxes for free, which
is tedious to argue in prose and obvious in a picture. So that the picture can
be trusted, every boxed panel here is a QC debug render exactly as
``qc.render_debug_frame`` wrote it — this script only crops, scales and
captions. It draws no boxes of its own, and nothing it emits can disagree with
what conversion produced.

Frame geometry in the captions is read from the canonical annotations for the
same reason: the numbers are quoted, never recomputed.

    python scripts/make_demo_assets.py \
        --run-dir data/raw/run_0001 \
        --qc-dir data/qc/run_0001/debug \
        --annotations data/datasets/v001/annotations/run_0001.json \
        --out-dir assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (23, 26, 31)
PAPER = (255, 255, 255)
GUTTER = 14
CAPTION_H = 40

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_annotations(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    return {a["frame_index"]: a for a in data}


def _frame_name(annotation: dict, index: int) -> str:
    # Annotations carry the source filename; fall back to the ingest convention.
    return annotation.get("normal") or f"frame_{index:06d}.png"


def _open(path: Path) -> Image.Image:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return Image.open(path).convert("RGB")


def _panel(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.LANCZOS)


def _crop_around(
    frame: Image.Image, box: dict, crop_w: int
) -> tuple[Image.Image, tuple[int, int]]:
    """A 16:9 window centred on the subject, clamped inside the frame."""
    crop_w = min(crop_w, frame.width)
    crop_h = min(round(crop_w * 9 / 16), frame.height)
    cx = box["x"] + box["w"] / 2
    cy = box["y"] + box["h"] / 2
    ox = min(max(round(cx - crop_w / 2), 0), frame.width - crop_w)
    oy = min(max(round(cy - crop_h / 2), 0), frame.height - crop_h)
    return frame.crop((ox, oy, ox + crop_w, oy + crop_h)), (ox, oy)


def _crop_like(frame: Image.Image, offset: tuple[int, int], size: tuple[int, int]):
    """The identical window on another pass of the same frame."""
    ox, oy = offset
    return frame.crop((ox, oy, ox + size[0], oy + size[1]))


def _compose(panels: list[tuple[Image.Image, str]]) -> Image.Image:
    """Lay panels out in a row, each with a caption underneath."""
    width = sum(p.width for p, _ in panels) + GUTTER * (len(panels) - 1)
    height = max(p.height for p, _ in panels) + CAPTION_H
    sheet = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(sheet)
    font = _font(17)

    x = 0
    for panel, caption in panels:
        sheet.paste(panel, (x, 0))
        draw.text((x + 2, panel.height + 11), caption, fill=INK, font=font)
        x += panel.width + GUTTER
    return sheet


def make_triptych(
    run_dir: Path,
    qc_dir: Path,
    run_id: str,
    annotation: dict,
    index: int,
    panel_width: int,
    crop_w: int,
) -> Image.Image:
    """normal | mask | the pipeline's own debug render — why labels are free."""
    name = _frame_name(annotation, index)
    normal = _open(run_dir / "normal" / name)
    mask = _open(run_dir / "mask" / name)
    debug = _open(qc_dir / f"{run_id}_{index:06d}.png")
    box = annotation["boxes"][0]

    # Full frames put the drone at a twelfth of the panel width, too small to
    # read the silhouette against; crop in so the mask panel earns its place.
    normal_crop, offset = _crop_around(normal, box, crop_w)
    window = normal_crop.size
    mask_crop = _crop_like(mask, offset, window)
    debug_crop = _crop_like(debug, offset, window)

    return _compose(
        [
            (_panel(normal_crop, panel_width), "1 · normal render — what a detector sees"),
            (_panel(mask_crop, panel_width), "2 · mask pass — the same frame, drone isolated"),
            (
                _panel(debug_crop, panel_width),
                f"3 · QC debug render — {box['w']}x{box['h']} px, no annotator",
            ),
        ]
    )


def make_scale_strip(
    qc_dir: Path,
    run_id: str,
    annotations: dict[int, dict],
    indices: list[int],
    panel_width: int,
    crop_w: int,
) -> Image.Image:
    """The same flight at four distances, cropped through an identical window.

    One run spans an order of magnitude of object scale; a reviewer should not
    have to take that on faith.
    """
    panels = []
    for index in indices:
        annotation = annotations[index]
        box = annotation["boxes"][0]
        debug = _open(qc_dir / f"{run_id}_{index:06d}.png")
        crop, _ = _crop_around(debug, box, crop_w)
        panels.append(
            (_panel(crop, panel_width), f"frame {index} · {box['w']}x{box['h']} px")
        )
    return _compose(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="raw run: normal/, mask/")
    parser.add_argument("--qc-dir", type=Path, required=True, help="qc/{run_id}/debug")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    parser.add_argument("--run-id", default=None, help="defaults to the run dir name")
    parser.add_argument("--frame", type=int, default=0, help="hero frame index")
    parser.add_argument(
        "--strip",
        default="0,50,100,147",
        help="comma-separated frame indices for the scale strip",
    )
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--hero-crop", type=int, default=1000)
    parser.add_argument("--strip-panel-width", type=int, default=300)
    parser.add_argument("--strip-crop", type=int, default=640)
    args = parser.parse_args()

    run_id = args.run_id or args.run_dir.name
    annotations = load_annotations(args.annotations)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    triptych = make_triptych(
        args.run_dir,
        args.qc_dir,
        run_id,
        annotations[args.frame],
        args.frame,
        args.panel_width,
        args.hero_crop,
    )
    triptych_path = args.out_dir / "label-derivation.png"
    triptych.save(triptych_path, optimize=True)
    print(f"{triptych_path} {triptych.size[0]}x{triptych.size[1]}")

    indices = [int(i) for i in args.strip.split(",")]
    strip = make_scale_strip(
        args.qc_dir,
        run_id,
        annotations,
        indices,
        args.strip_panel_width,
        args.strip_crop,
    )
    strip_path = args.out_dir / "scale-strip.png"
    strip.save(strip_path, optimize=True)
    print(f"{strip_path} {strip.size[0]}x{strip.size[1]}")


if __name__ == "__main__":
    main()

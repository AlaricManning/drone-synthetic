"""Command-line entrypoints: ingest, convert, submit."""

import argparse
import sys
from datetime import date
from pathlib import Path

from dronesynth.config import load_convert_config
from dronesynth.datagen.convert import convert_run
from dronesynth.ingest.capture import ingest_capture


def _ingest(args: argparse.Namespace) -> int:
    config = load_convert_config(args.config)
    result = ingest_capture(
        normal_root=args.normal,
        mask_root=args.mask,
        run_id=args.run_id,
        raw_root=args.raw_root or config.storage.raw_root,
        captured_at=args.captured_at,
        ue_map=args.ue_map,
        drone_model=args.drone_model,
    )
    manifest = result.manifest
    print(f"registered {manifest.run_id}: {manifest.frame_count} frame pairs")
    print(f"  sequence: {manifest.camera_sequence} ({manifest.ue_map}, {manifest.drone_model})")
    print(f"  location: {result.location}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    config = load_convert_config(args.config)
    result = convert_run(
        run_id=args.run_id,
        config=config,
        dataset_version=args.version,
    )
    report = result.report
    print(f"run {report.run_id}: {report.frames} frames, {report.total_boxes} boxes, "
          f"{report.empty_frames} empty frames")
    if report.total_boxes:
        print(f"  box area {report.box_area_min}..{report.box_area_max} px, "
              f"fill ratio {report.fill_ratio_min}..{report.fill_ratio_max}")
    print(f"  dataset: {result.dataset_location}")
    print(f"  qc:      {result.qc_location}")
    if report.flags:
        print(f"  {len(report.flags)} flag(s) — review these frames in the debug folder:")
        for flag in report.flags:
            print(f"    frame {flag.frame_index}: {flag.reason}")
    else:
        print("  no flags")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dronesynth",
        description="UE5/EasySynth paired renders -> versioned YOLO datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="validate a completed capture, write its manifest, sync to raw storage",
    )
    ingest.add_argument("--config", type=Path, required=True, help="config YAML (storage roots)")
    ingest.add_argument("--normal", type=Path, required=True, help="normal render directory")
    ingest.add_argument("--mask", type=Path, required=True, help="mask render directory")
    ingest.add_argument("--run-id", required=True, help="run id to register, e.g. run_0001")
    ingest.add_argument("--ue-map", required=True, help="UE map/level the capture was rendered in")
    ingest.add_argument("--drone-model", required=True, help="drone asset rendered in the capture")
    ingest.add_argument(
        "--captured-at",
        default=date.today().isoformat(),
        help="ISO date of the render session (default: today)",
    )
    ingest.add_argument(
        "--raw-root",
        default=None,
        help="override storage.raw_root from config, e.g. s3://bucket/raw",
    )

    convert = subparsers.add_parser(
        "convert",
        help="convert a raw run into a versioned dataset (annotations, YOLO export, QC)",
    )
    convert.add_argument("--config", type=Path, required=True, help="conversion config YAML")
    convert.add_argument("--run-id", required=True, help="registered run to convert, e.g. run_0001")
    convert.add_argument("--version", required=True, help="dataset version to write, e.g. v001")

    subparsers.add_parser(
        "submit",
        help="submit a conversion job to AWS Batch",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "convert":
        return _convert(args)
    print(f"'{args.command}' is not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

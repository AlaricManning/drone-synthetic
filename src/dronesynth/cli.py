"""Command-line entrypoints: ingest, convert, submit."""

import argparse
import sys
from pathlib import Path

from dronesynth.config import load_convert_config
from dronesynth.datagen.convert import convert_run


def _convert(args: argparse.Namespace) -> int:
    config = load_convert_config(args.config)
    result = convert_run(
        run_id=args.run_id,
        normal_root=args.normal,
        mask_root=args.mask,
        config=config,
        dataset_version=args.version,
    )
    report = result.report
    print(f"run {report.run_id}: {report.frames} frames, {report.total_boxes} boxes, "
          f"{report.empty_frames} empty frames")
    if report.total_boxes:
        print(f"  box area {report.box_area_min}..{report.box_area_max} px, "
              f"fill ratio {report.fill_ratio_min}..{report.fill_ratio_max}")
    print(f"  dataset: {result.dataset_dir}")
    print(f"  qc:      {result.qc_dir}")
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

    subparsers.add_parser(
        "ingest",
        help="validate a completed capture, write its manifest, sync to raw storage",
    )

    convert = subparsers.add_parser(
        "convert",
        help="convert a raw run into a versioned dataset (annotations, YOLO export, QC)",
    )
    convert.add_argument("--config", type=Path, required=True, help="conversion config YAML")
    convert.add_argument("--normal", type=Path, required=True, help="normal render directory")
    convert.add_argument("--mask", type=Path, required=True, help="mask render directory")
    convert.add_argument("--run-id", required=True, help="run id, e.g. run_0001")
    convert.add_argument("--version", required=True, help="dataset version to write, e.g. v001")

    subparsers.add_parser(
        "submit",
        help="submit a conversion job to AWS Batch",
    )

    args = parser.parse_args()
    if args.command == "convert":
        return _convert(args)
    print(f"'{args.command}' is not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

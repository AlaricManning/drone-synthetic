"""Command-line entrypoints: ingest, convert, submit."""

import argparse
import sys


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
    subparsers.add_parser(
        "convert",
        help="convert a raw run into a versioned dataset (annotations, YOLO export, QC)",
    )
    subparsers.add_parser(
        "submit",
        help="submit a conversion job to AWS Batch",
    )

    args = parser.parse_args()
    print(f"'{args.command}' is not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

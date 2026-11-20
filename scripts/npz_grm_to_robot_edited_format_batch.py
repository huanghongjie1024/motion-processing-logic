#!/usr/bin/env python3
"""Convert a folder of GMR NPZ motions to the robot motion editor format.

Reuses the adjacent single-file converter. No retargeting or visualization is
performed. Recursive mode preserves subdirectories and excludes the output tree.
"""

import argparse
import contextlib
import io
import math
import os
from pathlib import Path
import sys
import tempfile

from npz_grm_to_robot_edited_format import convert_npz


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_folder", required=True, type=Path, help="Input NPZ directory.")
    parser.add_argument("--tgt_folder", required=True, type=Path, help="Output NPZ directory.")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--quat_order", choices=["xyzw", "wxyz"], default="xyzw",
                        help="Source quaternion order; GMR root_rot defaults to xyzw.")
    parser.add_argument("--framerate", type=float, default=None,
                        help="Override FPS metadata (no resampling); otherwise preserve input FPS.")
    parser.add_argument("--joint_names_file", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="Show single-file conversion details.")
    return parser


def convert_one(source, destination, args):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The original converter expects an NPZ path. Replace the final file only
    # after successful conversion, so interrupted runs do not leave partial NPZs.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.stem}.",
        suffix=".npz", delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        output_context = (contextlib.nullcontext() if args.verbose
                          else contextlib.redirect_stdout(io.StringIO()))
        with output_context:
            convert_npz(
                input_npz=str(source), output_npz=str(temporary_path),
                quat_order=args.quat_order, framerate=args.framerate,
                joint_names_file=args.joint_names_file, urdf=args.urdf,
                verbose=args.verbose,
            )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    source_root = args.src_folder.expanduser().resolve()
    target_root = args.tgt_folder.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"Input directory does not exist: {source_root}")
    if target_root == source_root:
        parser.error("Input and output directories must differ")
    if target_root.exists() and not target_root.is_dir():
        parser.error(f"Output path is not a directory: {target_root}")
    if args.framerate is not None and (
        not math.isfinite(args.framerate) or args.framerate <= 0
    ):
        parser.error("--framerate must be finite and positive")
    for name in ("joint_names_file", "urdf"):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f"--{name} file does not exist: {value}")
            setattr(args, name, value)

    sources = []
    for directory, subdirs, filenames in os.walk(source_root, followlinks=False):
        # Exclude existing outputs even when the output folder is inside input.
        subdirs[:] = sorted(
            name for name in subdirs
            if (Path(directory) / name).resolve() != target_root
        ) if args.recursive else []
        for name in sorted(filenames):
            source = Path(directory) / name
            if source.suffix.lower() == ".npz" and source.is_file():
                sources.append(source)
    if not sources:
        parser.error(f"No NPZ files found in {source_root}")

    jobs = {}
    for source in sources:
        destination = (target_root / source.relative_to(source_root)).with_suffix(".npz")
        if destination in jobs:
            parser.error(f"Output collision: {jobs[destination]} and {source}")
        if destination.resolve() == source.resolve():
            parser.error(f"Output resolves to its input file: {source}")
        jobs[destination] = source

    succeeded = skipped = 0
    failures = []
    for index, (destination, source) in enumerate(jobs.items(), start=1):
        if destination.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(jobs)}] Skipped: {destination}")
            continue
        try:
            convert_one(source, destination, args)
            succeeded += 1
            print(f"[{index}/{len(jobs)}] Saved: {destination}")
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"[{index}/{len(jobs)}] Failed: {source}: {exc}", file=sys.stderr)

    print(f"Finished: {succeeded} succeeded, {skipped} skipped, {len(failures)} failed.")
    for source, error in failures:
        print(f"  {source}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

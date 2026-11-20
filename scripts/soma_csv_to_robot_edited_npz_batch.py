#!/usr/bin/env python3
"""Batch-convert SOMA Retargeter CSV files to Robot Motion Editor NPZ files.

SOMA CSV stores root translation in centimetres and root/joint rotations in
degrees. Robot Motion Editor expects metres, radians, and root quaternions in
[w, x, y, z] order. By default, 120 Hz SOMA data is resampled to 30 Hz.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_folder", required=True, type=Path,
                        help="Folder containing SOMA CSV files.")
    parser.add_argument("--tgt_folder", required=True, type=Path,
                        help="Folder for Robot Motion Editor NPZ files.")
    parser.add_argument("--recursive", action="store_true",
                        help="Include CSV files in subdirectories and preserve their layout.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace NPZ files that already exist.")
    parser.add_argument("--source_fps", type=float, default=120.0,
                        help="Actual CSV sampling rate; CSV has no FPS field (default: 120).")
    parser.add_argument("--target_fps", type=float, default=30.0,
                        help="Output sampling rate (default: 30).")
    return parser


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
    return [name.strip() for name in header]


def target_sample_times(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if num_frames < 1:
        raise ValueError("motion contains no frames")
    if num_frames == 1:
        return np.zeros(1, dtype=np.float64)
    duration = (num_frames - 1) / source_fps
    output_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    return np.arange(output_count, dtype=np.float64) / target_fps


def interpolate_columns(values: np.ndarray, source_times: np.ndarray,
                        output_times: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(output_times, source_times, values[:, column])
        for column in range(values.shape[1])
    ])


def convert_csv(source: Path, destination: Path, source_fps: float,
                target_fps: float) -> tuple[int, int]:
    header = read_header(source)
    if header[:7] != ROOT_COLUMNS:
        raise ValueError(
            "unexpected SOMA header; the first seven columns must be: "
            + ", ".join(ROOT_COLUMNS)
        )
    joint_names = header[7:]
    if not joint_names:
        raise ValueError("CSV contains no actuated joint columns")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("CSV contains duplicate joint names")

    values = np.loadtxt(source, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
    if values.shape[0] == 0:
        raise ValueError("CSV contains no motion frames")
    if values.shape[1] != len(header):
        raise ValueError(
            f"header has {len(header)} columns but motion rows have {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("CSV contains NaN or infinite values")

    input_count = values.shape[0]
    source_times = np.arange(input_count, dtype=np.float64) / source_fps
    output_times = target_sample_times(input_count, source_fps, target_fps)

    base_pos_w = interpolate_columns(values[:, 1:4] * 0.01, source_times, output_times)
    joint_pos = interpolate_columns(np.deg2rad(values[:, 7:]), source_times, output_times)
    root_rotations = Rotation.from_euler("xyz", values[:, 4:7], degrees=True)
    if input_count == 1:
        quat_xyzw = root_rotations.as_quat()
    else:
        quat_xyzw = Slerp(source_times, root_rotations)(output_times).as_quat()
    base_quat_w = quat_xyzw[:, [3, 0, 1, 2]]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(
            temporary,
            joint_pos=joint_pos.astype(np.float32),
            base_pos_w=base_pos_w.astype(np.float32),
            base_quat_w=base_quat_w.astype(np.float32),
            joint_names=np.asarray(joint_names, dtype=str),
            framerate=np.asarray(target_fps, dtype=np.float32),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return input_count, len(output_times)


def collect_sources(source_root: Path, target_root: Path, recursive: bool) -> list[Path]:
    sources: list[Path] = []
    for directory, subdirs, filenames in os.walk(source_root, followlinks=False):
        if recursive:
            subdirs[:] = sorted(
                name for name in subdirs
                if (Path(directory) / name).resolve() != target_root
            )
        else:
            subdirs[:] = []
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() == ".csv" and path.is_file():
                sources.append(path)
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_root = args.src_folder.expanduser().resolve()
    target_root = args.tgt_folder.expanduser().resolve()

    if not source_root.is_dir():
        parser.error(f"input directory does not exist: {source_root}")
    if source_root == target_root:
        parser.error("input and output directories must differ")
    if target_root.exists() and not target_root.is_dir():
        parser.error(f"output path is not a directory: {target_root}")
    for name in ("source_fps", "target_fps"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.target_fps > args.source_fps:
        parser.error("--target_fps cannot exceed --source_fps")

    sources = collect_sources(source_root, target_root, args.recursive)
    if not sources:
        parser.error(f"no CSV files found in {source_root}")

    succeeded = skipped = 0
    failures: list[tuple[Path, str]] = []
    for index, source in enumerate(sources, start=1):
        destination = (target_root / source.relative_to(source_root)).with_suffix(".npz")
        if destination.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(sources)}] Skipped: {destination}")
            continue
        try:
            input_count, output_count = convert_csv(
                source, destination, args.source_fps, args.target_fps
            )
            succeeded += 1
            print(
                f"[{index}/{len(sources)}] Saved: {destination} | "
                f"{input_count}@{args.source_fps:g} -> "
                f"{output_count}@{args.target_fps:g} Hz"
            )
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"[{index}/{len(sources)}] Failed: {source}: {exc}", file=sys.stderr)

    print(f"Finished: {succeeded} succeeded, {skipped} skipped, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

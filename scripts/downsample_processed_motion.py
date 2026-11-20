#!/usr/bin/env python3
"""Downsample already processed motion NPZ/PKL files on one shared time axis.

This is intended for legacy files produced from 120 Hz BVH motions without
frame decimation, even when their stored fps/framerate field already says 30.
NPZ remains NPZ, PKL remains PKL, and every per-frame field is sliced with the
same indices. This script does not interpolate or modify joint/pose values.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path
import sys
import tempfile

import numpy as np


MOTION_KEYS = (
    "joint_pos", "dof_pos", "qpos", "root_pos", "base_pos_w",
    "root_rot", "root_quat", "base_quat_w", "base_quat",
)
KNOWN_FRAME_KEYS = {
    "joint_pos", "dof_pos", "qpos", "joint_vel", "dof_vel", "qvel",
    "root_pos", "base_pos_w", "root_rot", "root_quat", "base_quat_w",
    "base_quat", "root_vel", "root_ang_vel", "base_lin_vel",
    "base_ang_vel", "contact_mask", "root_z_correction", "local_body_pos",
    "human_joints",
}
NON_FRAME_KEYS = {
    "fps", "framerate", "joint_names", "link_body_list", "body_names",
    "grounding_marker_frames", "grounding_marker_corrections",
    "grounding_window_seconds", "foot_ground_height", "left_foot_body",
    "right_foot_body", "num_frames",
}


def downsample_frame_indices(num_frames, source_fps, target_fps):
    if num_frames < 1:
        raise ValueError("Cannot downsample an empty motion.")
    if target_fps > source_fps:
        raise ValueError("Target FPS cannot exceed source FPS.")
    if math.isclose(source_fps, target_fps, rel_tol=1e-6) or num_frames == 1:
        return np.arange(num_frames, dtype=np.int64)
    duration = (num_frames - 1) / source_fps
    output_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    indices = np.rint(
        np.arange(output_count, dtype=np.float64) * source_fps / target_fps
    ).astype(np.int64)
    indices = np.clip(indices, 0, num_frames - 1)
    if np.any(np.diff(indices) <= 0):
        raise RuntimeError("Downsampling produced duplicate or unordered indices.")
    return indices


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One .npz/.pkl/.pickle file.")
    inputs.add_argument("--src_folder", type=Path, help="Batch input directory.")
    parser.add_argument("--output", type=Path, help="Single-file output path.")
    parser.add_argument("--tgt_folder", type=Path, help="Batch output directory.")
    parser.add_argument("--source_fps", type=float, default=120,
                        help="Actual sampling rate of existing frames (default: 120).")
    parser.add_argument("--target_fps", type=float, default=30,
                        help="Output sampling rate (default: 30).")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate and report frame counts without writing files.")
    parser.add_argument("--protocol", type=int, default=4, help="Pickle protocol (default: 4).")
    return parser


def _load_motion(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            keys = list(archive.files)
            if len(keys) == 1:
                value = archive[keys[0]]
                if value.shape == () and value.dtype == object:
                    obj = value.item()
                    if isinstance(obj, dict):
                        return dict(obj), "object_npz"
            return {key: archive[key] for key in keys}, "npz"
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"PKL root object must be a dict, got {type(data).__name__}")
        return dict(data), "pickle"
    raise ValueError(f"Unsupported extension: {path.suffix}")


def _frame_count(data):
    counts = []
    for key in MOTION_KEYS:
        if key not in data or data[key] is None:
            continue
        array = np.asarray(data[key])
        if array.ndim >= 1:
            counts.append((key, array.shape[0]))
    if not counts:
        raise KeyError("No motion array found; expected joint_pos/dof_pos/root_pos/base_pos_w etc.")
    unique = {count for _, count in counts}
    if len(unique) != 1:
        raise ValueError(f"Core motion frame counts differ: {dict(counts)}")
    count = unique.pop()
    if count < 1:
        raise ValueError("Motion contains no frames.")
    return count


def _slice_like(value, indices):
    array = np.asarray(value)
    sliced = array[indices]
    if isinstance(value, np.ndarray):
        return sliced
    if isinstance(value, tuple):
        return tuple(sliced.tolist())
    if isinstance(value, list):
        return sliced.tolist()
    return sliced


def _fps_value(original, target_fps):
    if isinstance(original, np.ndarray):
        return np.asarray(target_fps, dtype=original.dtype)
    if isinstance(original, np.generic):
        return type(original)(target_fps)
    if isinstance(original, int) and float(target_fps).is_integer():
        return int(target_fps)
    return float(target_fps)


def downsample_motion(data, source_fps, target_fps):
    frame_count = _frame_count(data)
    indices = downsample_frame_indices(frame_count, source_fps, target_fps)
    output = {}
    sliced_keys = []

    for key, value in data.items():
        if key in {"fps", "framerate"}:
            output[key] = _fps_value(value, target_fps)
            continue
        if key == "num_frames":
            output[key] = _fps_value(value, len(indices))
            continue
        if key == "grounding_marker_frames":
            original = np.asarray(value)
            scaled = original.astype(np.float64) * (target_fps / source_fps)
            output[key] = scaled if isinstance(value, np.ndarray) else scaled.tolist()
            continue
        if value is None or key in NON_FRAME_KEYS:
            output[key] = value
            continue

        array = np.asarray(value)
        if key in KNOWN_FRAME_KEYS and (array.ndim < 1 or array.shape[0] != frame_count):
            raise ValueError(
                f"Known per-frame field '{key}' has shape {array.shape}; "
                f"expected first dimension {frame_count}."
            )
        if array.ndim >= 1 and array.shape[0] == frame_count:
            output[key] = _slice_like(value, indices)
            sliced_keys.append(key)
        else:
            output[key] = value

    if "fps" not in output and "framerate" not in output:
        output["framerate" if "joint_pos" in output else "fps"] = float(target_fps)
    return output, indices, sliced_keys


def _atomic_save(path, data, style, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        suffix = ".npz" if style in {"npz", "object_npz"} else ".tmp"
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=suffix, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if style == "npz":
                np.savez(handle, **data)
            elif style == "object_npz":
                np.savez(handle, data=data)
            else:
                pickle.dump(data, handle, protocol=protocol)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def convert_one(source, destination, args):
    data, style = _load_motion(source)
    output, indices, sliced_keys = downsample_motion(
        data, args.source_fps, args.target_fps
    )
    if not args.dry_run:
        _atomic_save(destination, output, style, args.protocol)
    return _frame_count(data), len(indices), sliced_keys


def _validate_args(parser, args):
    for name in ("source_fps", "target_fps"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.target_fps > args.source_fps:
        parser.error("--target_fps cannot exceed --source_fps")
    if args.protocol < 0 or args.protocol > pickle.HIGHEST_PROTOCOL:
        parser.error(f"--protocol must be between 0 and {pickle.HIGHEST_PROTOCOL}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.input is not None:
        if args.tgt_folder is not None or args.recursive or args.overwrite:
            parser.error("--tgt_folder, --recursive and --overwrite require --src_folder")
        if args.output is None and not args.dry_run:
            parser.error("--output is required unless --dry_run is used")
        source = args.input.expanduser().resolve()
        if not source.is_file():
            parser.error(f"Input file does not exist: {source}")
        destination = args.output.expanduser().resolve() if args.output else source
        if not args.dry_run and destination.suffix.lower() != source.suffix.lower():
            parser.error("Input and output extensions must match")
        if not args.dry_run and destination == source:
            parser.error("Input and output files must differ")
        old_count, new_count, keys = convert_one(source, destination, args)
        action = "Would save" if args.dry_run else "Saved"
        print(f"{action}: {source} -> {destination}")
        print(f"Frames: {old_count} @ {args.source_fps:g} Hz -> {new_count} @ {args.target_fps:g} Hz")
        print("Sliced fields: " + ", ".join(keys))
        return 0

    if args.output is not None:
        parser.error("Use --tgt_folder with --src_folder, not --output")
    if args.tgt_folder is None and not args.dry_run:
        parser.error("--tgt_folder is required unless --dry_run is used")
    source_root = args.src_folder.expanduser().resolve()
    target_root = args.tgt_folder.expanduser().resolve() if args.tgt_folder else None
    if not source_root.is_dir():
        parser.error(f"Input directory does not exist: {source_root}")
    if target_root == source_root:
        parser.error("Input and output directories must differ")

    jobs = []
    for directory, subdirs, filenames in os.walk(source_root, followlinks=False):
        subdirs[:] = sorted(
            name for name in subdirs
            if target_root is None or (Path(directory) / name).resolve() != target_root
        ) if args.recursive else []
        for name in sorted(filenames):
            source = Path(directory) / name
            if source.suffix.lower() not in {".npz", ".pkl", ".pickle"}:
                continue
            destination = (
                target_root / source.relative_to(source_root)
                if target_root is not None else source
            )
            jobs.append((source, destination))
    if not jobs:
        parser.error(f"No NPZ/PKL files found in {source_root}")

    succeeded = skipped = 0
    failures = []
    for index, (source, destination) in enumerate(jobs, start=1):
        if not args.dry_run and destination.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(jobs)}] Skipped: {destination}")
            continue
        try:
            old_count, new_count, keys = convert_one(source, destination, args)
            succeeded += 1
            verb = "Would convert" if args.dry_run else "Saved"
            print(
                f"[{index}/{len(jobs)}] {verb}: {source} -> {destination} | "
                f"{old_count}@{args.source_fps:g} -> {new_count}@{args.target_fps:g} | "
                f"fields={','.join(keys)}"
            )
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"[{index}/{len(jobs)}] Failed: {source}: {exc}", file=sys.stderr)

    print(f"Finished: {succeeded} converted, {skipped} skipped, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

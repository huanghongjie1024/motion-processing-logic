#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .npz/.npy motion file to .pkl.

支持：
1. 普通 npz:
   np.savez("a.npz", fps=..., root_pos=..., root_rot=..., dof_pos=...)

2. 对象字典 npz:
   np.savez("a.npz", data=dict_obj)

3. npy 字典:
   np.save("a.npy", dict_obj)

默认输出 pkl 内容是 dict：
{
    "fps": ...,
    "root_pos": ...,
    "root_rot": ...,
    "dof_pos": ...,
    ...
}

用法：
python npz_to_pkl.py --input walk.npz --output walk.pkl
python npz_to_pkl.py --input walk_www.npz --output walk_www.pkl --rename_to_original_keys
"""

from __future__ import print_function

import argparse
import os
import pickle
from pathlib import Path
import sys
import tempfile

import numpy as np


def _is_npz_file(data):
    return hasattr(data, "files") and hasattr(data, "__getitem__")


def _unwrap_single_object_dict(data):
    """Support np.savez(path, data=dict_obj) and np.save(path, dict_obj)."""
    if isinstance(data, dict):
        return data

    if _is_npz_file(data):
        keys = list(data.files)

        # np.savez(path, data=dict_obj)
        if len(keys) == 1:
            arr = data[keys[0]]
            if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
                obj = arr.item()
                if isinstance(obj, dict):
                    return obj

        # 普通多 key npz
        return {k: data[k] for k in keys}

    # np.save(path, dict_obj)
    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        obj = data.item()
        if isinstance(obj, dict):
            return obj

    raise TypeError("Unsupported input data type: {}".format(type(data)))


def load_npz_or_npy(path):
    raw = np.load(path, allow_pickle=True)
    try:
        return _unwrap_single_object_dict(raw)
    finally:
        if _is_npz_file(raw):
            raw.close()


def rename_to_original_keys(data):
    """
    把 convert_motion_npz_format.py 输出格式转回常见原始 key 名：
      joint_pos   -> dof_pos
      base_pos_w  -> root_pos
      base_quat_w -> root_rot
      framerate   -> fps

    注意：这里只改 key 名，不改四元数顺序。
    如果你的 base_quat_w 是 [w,x,y,z]，而目标 pkl 需要 root_rot=[x,y,z,w]，
    请同时使用 --quat_wxyz_to_xyzw。
    """
    out = dict(data)

    mapping = {
        "joint_pos": "dof_pos",
        "base_pos_w": "root_pos",
        "base_quat_w": "root_rot",
        "framerate": "fps",
    }

    for old_key, new_key in mapping.items():
        if old_key in out:
            out[new_key] = out.pop(old_key)

    return out


def quat_wxyz_to_xyzw(data, quat_key="root_rot"):
    """
    root_rot: [w,x,y,z] -> [x,y,z,w]
    """
    out = dict(data)
    if quat_key not in out:
        raise KeyError("Cannot find quaternion key '{}' in data keys: {}".format(
            quat_key, list(out.keys())
        ))

    q = np.asarray(out[quat_key], dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("{} must have shape (N,4), got {}".format(quat_key, q.shape))

    q = q[:, [1, 2, 3, 0]]

    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Found near-zero quaternion; cannot normalize safely.")
    q = q / norm

    out[quat_key] = q.astype(np.float32)
    return out


def arrays_to_lists(data):
    """
    可选：把 numpy.ndarray 转成 list，减少不同 numpy 版本之间 pickle 加载问题。
    但读取 pkl 后需要自己 np.asarray(...) 转回来。
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def inspect_data(data):
    print("[INFO] keys:")
    for k, v in data.items():
        arr = np.asarray(v)
        print("  {}: shape={}, dtype={}".format(k, arr.shape, arr.dtype))


def convert_npz_to_pkl(
    input_path,
    output_path,
    rename_keys=False,
    quat_wxyz_to_xyzw_flag=False,
    as_lists=False,
    protocol=4,
    inspect=False,
):
    data = load_npz_or_npy(input_path)

    print("[INFO] input: {}".format(input_path))
    inspect_data(data)

    if rename_keys:
        data = rename_to_original_keys(data)
        print("[INFO] renamed keys to original-like format")
        inspect_data(data)

    if quat_wxyz_to_xyzw_flag:
        data = quat_wxyz_to_xyzw(data, quat_key="root_rot")
        print("[INFO] converted root_rot quaternion order: wxyz -> xyzw")

    if as_lists:
        data = arrays_to_lists(data)
        print("[INFO] converted numpy arrays to python lists")

    if inspect:
        print("[INFO] inspect only, no output saved.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.",
            suffix=".tmp", delete=False,
        ) as f:
            temporary_path = Path(f.name)
            pickle.dump(data, f, protocol=protocol)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print("[OK] saved: {}".format(output_path))
    print("[INFO] pickle protocol: {}".format(protocol))


def main(argv=None):
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Input .npz/.npy path")
    inputs.add_argument("--src_folder", type=Path, help="Batch input directory (.npz/.npy).")
    parser.add_argument("--output", default=None, help="Output .pkl path")
    parser.add_argument("--tgt_folder", type=Path, help="Batch output directory.")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing batch outputs.")

    parser.add_argument(
        "--rename_to_original_keys",
        action="store_true",
        help="Rename joint_pos/base_pos_w/base_quat_w/framerate to dof_pos/root_pos/root_rot/fps.",
    )
    parser.add_argument(
        "--quat_wxyz_to_xyzw",
        action="store_true",
        help="Convert root_rot quaternion order from [w,x,y,z] to [x,y,z,w]. Usually needed after --rename_to_original_keys.",
    )
    parser.add_argument(
        "--as_lists",
        action="store_true",
        help="Save arrays as Python lists for better cross-numpy compatibility.",
    )
    parser.add_argument(
        "--protocol",
        type=int,
        default=4,
        help="Pickle protocol. Use 4 for better compatibility with older Python.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print keys/shapes, do not save pkl.",
    )

    args = parser.parse_args(argv)

    if args.protocol < 0 or args.protocol > pickle.HIGHEST_PROTOCOL:
        parser.error(f"--protocol must be between 0 and {pickle.HIGHEST_PROTOCOL}")

    def convert(source, destination):
        convert_npz_to_pkl(
            input_path=source, output_path=destination,
            rename_keys=args.rename_to_original_keys,
            quat_wxyz_to_xyzw_flag=args.quat_wxyz_to_xyzw,
            as_lists=args.as_lists, protocol=args.protocol, inspect=args.inspect,
        )

    if args.src_folder is None:
        if args.tgt_folder is not None or args.recursive or args.overwrite:
            parser.error("--tgt_folder, --recursive and --overwrite require --src_folder")
        if not args.inspect and args.output is None:
            parser.error("--output is required unless --inspect is used")
        convert(args.input, args.output)
        return 0

    if args.output is not None:
        parser.error("Use --tgt_folder with --src_folder, not --output")
    if args.tgt_folder is None and not args.inspect:
        parser.error("--tgt_folder is required unless --inspect is used")
    source_root = args.src_folder.expanduser().resolve()
    target_root = args.tgt_folder.expanduser().resolve() if args.tgt_folder else None
    if not source_root.is_dir():
        parser.error(f"Input directory does not exist: {source_root}")
    if target_root == source_root:
        parser.error("Input and output directories must differ")
    if target_root is not None and target_root.exists() and not target_root.is_dir():
        parser.error(f"Output path is not a directory: {target_root}")

    jobs = []
    destinations = {}
    for directory, subdirs, filenames in os.walk(source_root, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs
                            if (Path(directory) / name).resolve() != target_root
                            ) if args.recursive else []
        for name in sorted(filenames):
            source = Path(directory) / name
            if source.suffix.lower() not in {".npz", ".npy"} or not source.is_file():
                continue
            destination = ((target_root / source.relative_to(source_root)).with_suffix(".pkl")
                           if target_root is not None else None)
            if destination is not None and not args.inspect:
                if destination in destinations:
                    parser.error(f"Output collision: {destinations[destination]} and {source}")
                destinations[destination] = source
            jobs.append((source, destination))
    if not jobs:
        parser.error(f"No NPZ/NPY files found in {source_root}")

    succeeded = skipped = 0
    failures = []
    for index, (source, destination) in enumerate(jobs, start=1):
        if not args.inspect and destination.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(jobs)}] Skipped: {destination}")
            continue
        try:
            print(f"[{index}/{len(jobs)}] Processing: {source}")
            convert(source, destination)
            succeeded += 1
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"Failed: {source}: {exc}", file=sys.stderr)
    action = "inspected" if args.inspect else "converted"
    print(f"Finished: {succeeded} {action}, {skipped} skipped, {len(failures)} failed.")
    for source, error in failures:
        print(f"  {source}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

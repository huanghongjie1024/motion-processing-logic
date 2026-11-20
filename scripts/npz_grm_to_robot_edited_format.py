#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert motion .npz/.npy/.dict-npz to the application format:

  joint_pos:   (num_frames, num_joints) float32
  base_pos_w:  (num_frames, 3)          float32, [x, y, z]
  base_quat_w: (num_frames, 4)          float32, [w, x, y, z]
  joint_names: (num_joints,)            string array, URDF joint names
  framerate:   scalar                   float32

Source keys supported:
  dof_pos / joint_pos / qpos             -> joint_pos
  root_pos / base_pos_w / base_pos       -> base_pos_w
  root_rot / root_quat / base_quat_w     -> base_quat_w
  fps / framerate                        -> framerate

This version supports:
  - normal npz files
  - npz/npy saved object dicts
  - joint names from the input data, a supplied name list, or a URDF
  - optional URDF parsing for revolute/continuous joint order
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _is_npz_file(data):
    return hasattr(data, "files") and hasattr(data, "__getitem__")


def _keys(data):
    if _is_npz_file(data):
        return list(data.files)
    if isinstance(data, dict):
        return list(data.keys())
    raise TypeError("Unsupported data type: {}".format(type(data)))


def _has_key(data, key):
    if _is_npz_file(data):
        return key in data.files
    if isinstance(data, dict):
        return key in data
    return False


def _get_value(data, key):
    if _is_npz_file(data) or isinstance(data, dict):
        return data[key]
    raise TypeError("Unsupported data type: {}".format(type(data)))


def _unwrap_single_object_dict(data):
    """Support np.savez(path, data=dict_obj) and np.save(path, dict_obj)."""
    if isinstance(data, dict):
        return data

    if _is_npz_file(data):
        keys = list(data.files)
        if len(keys) == 1:
            arr = data[keys[0]]
            if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
                obj = arr.item()
                if isinstance(obj, dict):
                    return obj
        return data

    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        obj = data.item()
        if isinstance(obj, dict):
            return obj

    return data


def load_motion_data(path):
    raw = np.load(path, allow_pickle=True)
    return _unwrap_single_object_dict(raw)


def _get_array(data, candidates, target_name):
    for key in candidates:
        if _has_key(data, key):
            arr = _get_value(data, key)
            if arr is not None:
                return np.asarray(arr)
    raise KeyError(
        "Cannot find source array for {}. Tried: {}. Available keys: {}".format(
            target_name, candidates, _keys(data)
        )
    )


def _get_optional_scalar(data, candidates):
    for key in candidates:
        if _has_key(data, key):
            val = _get_value(data, key)
            arr = np.asarray(val)
            if arr.shape == ():
                return float(arr.item())
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
    return None


def _load_names_from_file(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        names = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".npy":
        names = np.load(str(p), allow_pickle=True).tolist()
    else:
        names = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [str(x) for x in names]


def _load_names_from_urdf(urdf_path):
    root = ET.parse(str(urdf_path)).getroot()
    names = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type in ("revolute", "continuous"):
            names.append(joint.attrib["name"])
    return names


def _load_joint_names(path, num_joints, input_data=None, urdf=None):
    """Load joint names from a name list, URDF, or input data."""
    if path is not None:
        names = _load_names_from_file(path)
        source = "joint_names_file"
    elif urdf is not None:
        names = _load_names_from_urdf(urdf)
        source = "urdf"
    elif input_data is not None and _has_key(input_data, "joint_names"):
        names = np.asarray(_get_value(input_data, "joint_names"), dtype=str).tolist()
        source = "input_npz"
    else:
        raise ValueError(
            "Input has no joint_names. Supply --joint_names_file or --urdf."
        )

    if len(names) != num_joints:
        raise ValueError(
            "joint_names length mismatch from {}: got {}, but joint_pos has {} joints. "
            "Check the input joint_names, --joint_names_file, or --urdf.".format(
                source, len(names), num_joints
            )
        )

    print("[INFO] joint_names source: {}, count={}".format(source, len(names)))
    return np.asarray(names, dtype=str)


def _convert_quat_to_wxyz(quat, quat_order):
    quat = np.asarray(quat)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("Quaternion must have shape (N, 4), got {}".format(quat.shape))

    if quat_order == "xyzw":
        quat_wxyz = quat[:, [3, 0, 1, 2]]
    elif quat_order == "wxyz":
        quat_wxyz = quat.copy()
    else:
        raise ValueError("Unsupported quat_order: {}".format(quat_order))

    quat_wxyz = quat_wxyz.astype(np.float32)
    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Found near-zero quaternion; cannot normalize safely.")
    return quat_wxyz / norm


def convert_npz(
    input_npz,
    output_npz,
    quat_order="xyzw",
    framerate=None,
    joint_names_file=None,
    urdf=None,
    verbose=True,
):
    data = load_motion_data(input_npz)

    if verbose:
        print("[INFO] input: {}".format(input_npz))
        print("[INFO] available keys: {}".format(_keys(data)))

    joint_pos = _get_array(data, ["joint_pos", "dof_pos", "qpos"], "joint_pos").astype(np.float32)
    base_pos_w = _get_array(data, ["base_pos_w", "root_pos", "base_pos"], "base_pos_w").astype(np.float32)
    root_rot = _get_array(data, ["base_quat_w", "root_rot", "root_quat", "base_quat"], "base_quat_w")

    if joint_pos.ndim != 2:
        raise ValueError("joint_pos must have shape (N, J), got {}".format(joint_pos.shape))

    if _has_key(data, "base_quat_w") and quat_order == "xyzw":
        print("[WARN] input has key 'base_quat_w'; treating it as wxyz instead of xyzw.")
        quat_order = "wxyz"

    base_quat_w = _convert_quat_to_wxyz(root_rot.astype(np.float32), quat_order)
    joint_names = _load_joint_names(
        joint_names_file,
        joint_pos.shape[1],
        input_data=data,
        urdf=urdf,
    )

    if framerate is None:
        framerate = _get_optional_scalar(data, ["framerate", "fps"])
        if framerate is None:
            framerate = 30.0

    num_frames = joint_pos.shape[0]
    if base_pos_w.shape != (num_frames, 3):
        raise ValueError("base_pos_w must have shape ({}, 3), got {}".format(num_frames, base_pos_w.shape))
    if base_quat_w.shape != (num_frames, 4):
        raise ValueError("base_quat_w must have shape ({}, 4), got {}".format(num_frames, base_quat_w.shape))

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_path),
        joint_pos=joint_pos.astype(np.float32),
        base_pos_w=base_pos_w.astype(np.float32),
        base_quat_w=base_quat_w.astype(np.float32),
        joint_names=joint_names,
        framerate=np.asarray(framerate, dtype=np.float32),
    )

    print("[OK] Saved: {}".format(output_path))
    print("  joint_pos:   {}, {}".format(joint_pos.shape, joint_pos.dtype))
    print("  base_pos_w:  {}, {}".format(base_pos_w.shape, base_pos_w.dtype))
    print("  base_quat_w: {}, {}, order=[w,x,y,z]".format(base_quat_w.shape, base_quat_w.dtype))
    print("  joint_names: {}".format(joint_names.shape))
    print("  framerate:   {}".format(float(framerate)))


def inspect_npz(input_npz):
    data = load_motion_data(input_npz)
    print("[INFO] input: {}".format(input_npz))
    print("[INFO] data type: {}".format(type(data)))
    print("[INFO] keys:")
    for key in _keys(data):
        val = _get_value(data, key)
        arr = np.asarray(val)
        print("  {}: shape={}, dtype={}".format(key, arr.shape, arr.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .npz/.npy path")
    parser.add_argument("--output", default=None, help="Output .npz path")
    parser.add_argument(
        "--quat_order",
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion order in the source file. root_rot is often xyzw.",
    )
    parser.add_argument(
        "--framerate",
        type=float,
        default=None,
        help="Override output framerate. If omitted, use input framerate/fps or 30.",
    )
    parser.add_argument(
        "--joint_names_file",
        default=None,
        help="Optional txt/json/npy file containing URDF joint names. One name per line for txt.",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Optional URDF path. Revolute/continuous joint names are extracted in file order.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print keys/shapes and do not convert.",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_npz(args.input)
        return

    if args.output is None:
        raise ValueError("--output is required unless --inspect is used.")

    convert_npz(
        input_npz=args.input,
        output_npz=args.output,
        quat_order=args.quat_order,
        framerate=args.framerate,
        joint_names_file=args.joint_names_file,
        urdf=args.urdf,
    )


if __name__ == "__main__":
    main()

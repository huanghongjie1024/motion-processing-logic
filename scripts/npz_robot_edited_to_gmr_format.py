#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python npz_robot_edited_to_gmr_format.py --input edited.npz --output motion.npz --root_rot_order xyzw

Reverse converter for convert_motion_npz_format.py output.

Input application format:
  joint_pos:   (N, J)
  base_pos_w:  (N, 3)
  base_quat_w: (N, 4), order [w, x, y, z]
  joint_names: optional
  framerate:   scalar

Output original-like format:
  fps
  root_pos
  root_rot
  dof_pos

Default output root_rot order is xyzw, matching the common source format used by:
  python convert_motion_npz_format.py --quat_order xyzw
"""

from __future__ import print_function

import argparse
from pathlib import Path
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
        "Cannot find array for {}. Tried: {}. Available keys: {}".format(
            target_name, candidates, _keys(data)
        )
    )


def _get_optional_scalar(data, candidates, default=None):
    for key in candidates:
        if _has_key(data, key):
            arr = np.asarray(_get_value(data, key))
            if arr.shape == ():
                return float(arr.item())
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
    return default


def _normalize_quat_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError("Quaternion must have shape (N, 4), got {}".format(quat_wxyz.shape))

    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Found near-zero quaternion; cannot normalize safely.")
    return quat_wxyz / norm


def _convert_wxyz_to_order(quat_wxyz, order):
    quat_wxyz = _normalize_quat_wxyz(quat_wxyz)

    if order == "xyzw":
        # [w, x, y, z] -> [x, y, z, w]
        return quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    if order == "wxyz":
        return quat_wxyz.astype(np.float32)

    raise ValueError("Unsupported quaternion order: {}".format(order))


def _copy_template_extra(template_path, out_dict, num_frames):
    """
    Copy keys from an old/original npz as much as possible.
    Overwritten keys are not copied:
      fps/root_pos/root_rot/dof_pos/joint_pos/base_pos_w/base_quat_w/framerate
    """
    if template_path is None:
        return out_dict

    template = load_motion_data(template_path)
    skip_keys = set([
        "fps", "framerate",
        "root_pos", "base_pos_w", "base_pos",
        "root_rot", "root_quat", "base_quat_w", "base_quat",
        "dof_pos", "joint_pos", "qpos",
    ])

    for key in _keys(template):
        if key in skip_keys:
            continue

        val = np.asarray(_get_value(template, key))

        # 只对明显按帧存储的数组做帧数检查。比如 local_body_pos: (N, ...)
        if val.ndim >= 1 and val.shape[0] != num_frames:
            print("[WARN] skip extra key '{}': template first dim {} != num_frames {}".format(
                key, val.shape[0], num_frames
            ))
            continue

        out_dict[key] = val
        print("[INFO] copied extra key from template: {}, shape={}, dtype={}".format(
            key, val.shape, val.dtype
        ))

    return out_dict


def reverse_npz(
    input_npz,
    output_npz,
    root_rot_order="xyzw",
    fps=None,
    keep_joint_names=False,
    copy_extra_from=None,
    save_object_dict=False,
    verbose=True,
):
    data = load_motion_data(input_npz)

    if verbose:
        print("[INFO] input: {}".format(input_npz))
        print("[INFO] available keys: {}".format(_keys(data)))

    # 兼容标准输出格式，也兼容已经是类似原始格式的文件
    dof_pos = _get_array(data, ["joint_pos", "dof_pos", "qpos"], "dof_pos").astype(np.float32)
    root_pos = _get_array(data, ["base_pos_w", "root_pos", "base_pos"], "root_pos").astype(np.float32)
    base_quat_w = _get_array(data, ["base_quat_w", "root_rot", "root_quat", "base_quat"], "base_quat_w")

    if dof_pos.ndim != 2:
        raise ValueError("dof_pos/joint_pos must have shape (N, J), got {}".format(dof_pos.shape))

    num_frames = dof_pos.shape[0]
    if root_pos.shape != (num_frames, 3):
        raise ValueError("root_pos/base_pos_w must have shape ({}, 3), got {}".format(
            num_frames, root_pos.shape
        ))

    # 注意：convert_motion_npz_format.py 的输出 base_quat_w 固定是 [w,x,y,z]
    # 如果输入已经是 root_rot 但不是 base_quat_w，这里默认仍按 wxyz 处理。
    root_rot = _convert_wxyz_to_order(base_quat_w, root_rot_order)

    if fps is None:
        fps = _get_optional_scalar(data, ["framerate", "fps"], default=30.0)

    out = {
        "fps": np.asarray(fps, dtype=np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
    }

    if keep_joint_names and _has_key(data, "joint_names"):
        out["joint_names"] = np.asarray(_get_value(data, "joint_names"), dtype=str)

    out = _copy_template_extra(copy_extra_from, out, num_frames)

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_object_dict:
        # 生成只有一个 data 对象的 npz：np.savez(path, data=out)
        np.savez(str(output_path), data=out)
    else:
        # 生成普通 npz：keys 直接是 fps/root_pos/root_rot/dof_pos/...
        np.savez(str(output_path), **out)

    print("[OK] Saved: {}".format(output_path))
    print("  fps:      {}".format(float(fps)))
    print("  root_pos: {}, {}, order=[x,y,z]".format(out["root_pos"].shape, out["root_pos"].dtype))
    print("  root_rot: {}, {}, order={}".format(out["root_rot"].shape, out["root_rot"].dtype, root_rot_order))
    print("  dof_pos:  {}, {}".format(out["dof_pos"].shape, out["dof_pos"].dtype))
    print("  keys:     {}".format(list(out.keys())))


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
    parser.add_argument("--input", required=True, help="Input converted .npz/.npy path")
    parser.add_argument("--output", default=None, help="Output original-like .npz path")
    parser.add_argument(
        "--root_rot_order",
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion order for output root_rot. Default xyzw.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps. If omitted, use input framerate/fps or 30.",
    )
    parser.add_argument(
        "--keep_joint_names",
        action="store_true",
        help="Also save joint_names into output npz.",
    )
    parser.add_argument(
        "--copy_extra_from",
        default=None,
        help="Optional original/template npz. Copy extra keys like local_body_pos/link_body_list if frame count matches.",
    )
    parser.add_argument(
        "--save_object_dict",
        action="store_true",
        help="Save as np.savez(output, data=dict) instead of normal multi-key npz.",
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

    reverse_npz(
        input_npz=args.input,
        output_npz=args.output,
        root_rot_order=args.root_rot_order,
        fps=args.fps,
        keep_joint_names=args.keep_joint_names,
        copy_extra_from=args.copy_extra_from,
        save_object_dict=args.save_object_dict,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch-apply Robot Motion Editor ankle-based grounding to editor NPZ files.

For each frame, URDF forward kinematics computes the world positions of the
left and right ankle-roll joint origins.  The lower ankle is treated as the
supporting foot and the root Z coordinate is corrected using:

    estimated_sole_z = min(left_ankle_z, right_ankle_z) - sole_offset
    base_z_correction = ground_height - estimated_sole_z

The default ``always`` mode matches Robot Motion Editor's selected-region
automatic grounding: it raises penetration frames and lowers airborne frames.
Only ``base_pos_w[:, 2]`` is changed; all other NPZ arrays are preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class UrdfModel:
    root_link: str
    joints: tuple[UrdfJoint, ...]
    joint_by_name: dict[str, UrdfJoint]


def _parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"invalid URDF vector: {value!r}")
    return result


def load_urdf(path: Path) -> UrdfModel:
    root = ET.parse(path).getroot()
    links = {node.attrib["name"] for node in root.findall("link")}
    if not links:
        raise ValueError("URDF contains no links")

    raw_joints: list[UrdfJoint] = []
    children = set()
    for node in root.findall("joint"):
        name = node.attrib.get("name", "").strip()
        joint_type = node.attrib.get("type", "fixed").strip()
        parent_node = node.find("parent")
        child_node = node.find("child")
        if not name or parent_node is None or child_node is None:
            raise ValueError("URDF joint is missing name, parent, or child")
        parent = parent_node.attrib["link"]
        child = child_node.attrib["link"]
        origin = node.find("origin")
        xyz = _parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0))
        rpy = _parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0))
        axis_node = node.find("axis")
        axis = _parse_vector(axis_node.attrib.get("xyz") if axis_node is not None else None, (1, 0, 0))
        axis_norm = np.linalg.norm(axis)
        if joint_type in ("revolute", "continuous", "prismatic") and axis_norm <= 1.0e-12:
            raise ValueError(f"URDF joint {name} has a zero axis")
        if axis_norm > 1.0e-12:
            axis = axis / axis_norm
        raw_joints.append(UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=xyz,
            origin_rotation=Rotation.from_euler("xyz", rpy).as_matrix(),
            axis=axis,
        ))
        children.add(child)

    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link; found {roots}")

    # Topologically order joints from the root so FK never depends on XML order.
    pending = list(raw_joints)
    ordered: list[UrdfJoint] = []
    reached = {roots[0]}
    while pending:
        ready = [joint for joint in pending if joint.parent in reached]
        if not ready:
            names = ", ".join(joint.name for joint in pending)
            raise ValueError(f"URDF has a disconnected or cyclic joint graph: {names}")
        for joint in ready:
            ordered.append(joint)
            reached.add(joint.child)
            pending.remove(joint)

    return UrdfModel(
        root_link=roots[0],
        joints=tuple(ordered),
        joint_by_name={joint.name: joint for joint in ordered},
    )


def _validate_motion(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    required = ("joint_pos", "base_pos_w", "base_quat_w", "joint_names")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("NPZ is missing fields: " + ", ".join(missing))

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    base_pos = np.asarray(data["base_pos_w"], dtype=np.float64)
    base_quat = np.asarray(data["base_quat_w"], dtype=np.float64)
    joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"joint_pos shape {joint_pos.shape} does not match {len(joint_names)} joint names"
        )
    frame_count = joint_pos.shape[0]
    if base_pos.shape != (frame_count, 3):
        raise ValueError(f"base_pos_w must have shape ({frame_count}, 3), got {base_pos.shape}")
    if base_quat.shape != (frame_count, 4):
        raise ValueError(f"base_quat_w must have shape ({frame_count}, 4), got {base_quat.shape}")
    if frame_count < 1:
        raise ValueError("motion contains no frames")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names contains duplicates")
    if not all(np.isfinite(array).all() for array in (joint_pos, base_pos, base_quat)):
        raise ValueError("motion contains NaN or infinite values")
    norms = np.linalg.norm(base_quat, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("base_quat_w contains a zero quaternion")
    base_quat = base_quat / norms[:, None]
    return joint_pos, base_pos, base_quat, joint_names


def ankle_world_heights(
    model: UrdfModel,
    joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    joint_names: list[str],
    ankle_joint_names: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    for name in ankle_joint_names:
        if name not in model.joint_by_name:
            raise ValueError(f"URDF does not contain ankle joint: {name}")

    motion_index = {name: index for index, name in enumerate(joint_names)}
    unknown = sorted(name for name in joint_names if name not in model.joint_by_name)
    if unknown:
        raise ValueError("motion contains joints absent from URDF: " + ", ".join(unknown))

    frame_count = len(base_pos)
    # scipy expects xyzw while editor NPZ stores wxyz.
    root_rotation = Rotation.from_quat(base_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    link_position: dict[str, np.ndarray] = {model.root_link: base_pos}
    link_rotation: dict[str, np.ndarray] = {model.root_link: root_rotation}
    ankle_heights: dict[str, np.ndarray] = {}

    for joint in model.joints:
        parent_position = link_position[joint.parent]
        parent_rotation = link_rotation[joint.parent]
        origin_position = parent_position + np.einsum(
            "nij,j->ni", parent_rotation, joint.origin_xyz
        )
        origin_rotation = parent_rotation @ joint.origin_rotation
        ankle_heights[joint.name] = origin_position[:, 2]

        values = joint_pos[:, motion_index[joint.name]] if joint.name in motion_index else np.zeros(frame_count)
        if joint.joint_type in ("revolute", "continuous"):
            motion_rotation = Rotation.from_rotvec(values[:, None] * joint.axis).as_matrix()
            child_position = origin_position
            child_rotation = origin_rotation @ motion_rotation
        elif joint.joint_type == "prismatic":
            child_position = origin_position + np.einsum(
                "nij,nj->ni", origin_rotation, values[:, None] * joint.axis
            )
            child_rotation = origin_rotation
        elif joint.joint_type == "fixed":
            child_position = origin_position
            child_rotation = origin_rotation
        else:
            raise ValueError(f"unsupported URDF joint type {joint.joint_type!r}: {joint.name}")

        link_position[joint.child] = child_position
        link_rotation[joint.child] = child_rotation

    return tuple(ankle_heights[name] for name in ankle_joint_names)


def ground_motion(
    data: dict[str, np.ndarray],
    model: UrdfModel,
    ankle_joint_names: tuple[str, str],
    sole_offset_m: float,
    ground_height_m: float,
    mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    joint_pos, base_pos, base_quat, joint_names = _validate_motion(data)
    left_z, right_z = ankle_world_heights(
        model, joint_pos, base_pos, base_quat, joint_names, ankle_joint_names
    )
    estimated_sole_z = np.minimum(left_z, right_z) - sole_offset_m
    correction = ground_height_m - estimated_sole_z
    if mode == "raise_only":
        correction = np.maximum(correction, 0.0)

    grounded = {key: np.array(value, copy=True) for key, value in data.items()}
    grounded_base = np.array(data["base_pos_w"], copy=True)
    grounded_base[:, 2] += correction.astype(grounded_base.dtype, copy=False)
    grounded["base_pos_w"] = grounded_base
    residual = estimated_sole_z + correction - ground_height_m
    return grounded, {
        "frames": float(len(correction)),
        "changed_frames": float(np.count_nonzero(np.abs(correction) > 1.0e-9)),
        "min_correction_m": float(np.min(correction)),
        "max_correction_m": float(np.max(correction)),
        "max_abs_residual_m": float(np.max(np.abs(residual))),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def save_npz_atomic(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **data)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def collect_sources(source_root: Path, target_root: Path, recursive: bool) -> list[Path]:
    sources = []
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
            if path.suffix.lower() == ".npz" and path.is_file():
                sources.append(path)
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_folder", required=True, type=Path,
                        help="Folder containing Robot Motion Editor NPZ files.")
    parser.add_argument("--tgt_folder", required=True, type=Path,
                        help="Folder for grounded NPZ files.")
    parser.add_argument("--urdf", required=True, type=Path,
                        help="Path to the supplied robot URDF.")
    parser.add_argument("--recursive", action="store_true",
                        help="Include NPZ files in subdirectories and preserve their layout.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output NPZ files that already exist.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate and calculate corrections without writing files.")
    parser.add_argument("--left_ankle_joint", required=True)
    parser.add_argument("--right_ankle_joint", required=True)
    parser.add_argument("--sole_offset_m", required=True, type=float,
                        help="Vertical ankle-origin to sole distance in metres.")
    parser.add_argument("--ground_height_m", type=float, default=0.0,
                        help="Target ground-plane world Z in metres (default: 0).")
    parser.add_argument("--mode", choices=("always", "raise_only"), default="always",
                        help="always matches the editor; raise_only never lowers airborne frames.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_root = args.src_folder.expanduser().resolve()
    target_root = args.tgt_folder.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()

    if not source_root.is_dir():
        parser.error(f"input directory does not exist: {source_root}")
    if source_root == target_root:
        parser.error("input and output directories must differ")
    if target_root.exists() and not target_root.is_dir():
        parser.error(f"output path is not a directory: {target_root}")
    if not urdf_path.is_file():
        parser.error(f"URDF does not exist: {urdf_path}")
    if args.left_ankle_joint == args.right_ankle_joint:
        parser.error("left and right ankle joint names must differ")
    for name in ("sole_offset_m", "ground_height_m"):
        value = getattr(args, name)
        if not math.isfinite(value):
            parser.error(f"--{name} must be finite")
    if args.sole_offset_m < 0:
        parser.error("--sole_offset_m must be non-negative")

    try:
        model = load_urdf(urdf_path)
    except Exception as exc:
        parser.error(f"cannot load URDF {urdf_path}: {exc}")

    sources = collect_sources(source_root, target_root, args.recursive)
    if not sources:
        parser.error(f"no NPZ files found in {source_root}")

    print(f"Source folder: {source_root}")
    print(f"Target folder: {target_root}")
    print(f"URDF: {urdf_path}")
    print(f"Files found: {len(sources)}")

    succeeded = skipped = 0
    failures: list[tuple[Path, str]] = []
    for index, source in enumerate(sources, start=1):
        destination = target_root / source.relative_to(source_root)
        if destination.exists() and not args.overwrite and not args.dry_run:
            skipped += 1
            print(f"[{index}/{len(sources)}] Skipped: {destination}")
            continue
        try:
            data = load_npz(source)
            grounded, diagnostics = ground_motion(
                data,
                model,
                (args.left_ankle_joint, args.right_ankle_joint),
                args.sole_offset_m,
                args.ground_height_m,
                args.mode,
            )
            if not args.dry_run:
                save_npz_atomic(destination, grounded)
            succeeded += 1
            action = "Valid" if args.dry_run else "Saved"
            print(
                f"[{index}/{len(sources)}] {action}: {destination} | "
                f"frames={int(diagnostics['frames'])}, "
                f"changed={int(diagnostics['changed_frames'])}, "
                f"correction=[{diagnostics['min_correction_m']:.6f}, "
                f"{diagnostics['max_correction_m']:.6f}] m, "
                f"max_residual={diagnostics['max_abs_residual_m']:.3e} m"
            )
        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"[{index}/{len(sources)}] Failed: {source}: {exc}", file=sys.stderr)

    action = "validated" if args.dry_run else "grounded"
    print(f"Finished: {succeeded} {action}, {skipped} skipped, {len(failures)} failed.")
    if failures:
        print("Failed files:", file=sys.stderr)
        for path, message in failures:
            print(f"  {path}: {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

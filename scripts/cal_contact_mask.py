#!/usr/bin/env python3
"""用 BVH 前向运动学估计脚部高度、速度及左右脚接触掩码。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


class BVHNode:
    """BVH 层级节点"""
    def __init__(self, name, offset, channels):
        self.name = name
        self.offset = np.array(offset)
        self.channels = channels
        self.children = []
        self.parent = None


def read_bvh(filepath):
    """解析 BVH 文件，返回根节点、通道映射、帧数据与帧率"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    stack = []
    root = None
    node_map = {}

    # ---- 解析层级结构 ----
    while i < len(lines):
        line = lines[i]
        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[1]
            offset = []
            channels = []
            i += 1
            while not lines[i].startswith("OFFSET"):
                i += 1
            offset = list(map(float, lines[i].split()[1:]))
            i += 1
            if lines[i].startswith("CHANNELS"):
                parts = lines[i].split()
                n = int(parts[1])
                channels = parts[2:2 + n]
            node = BVHNode(name, offset, channels)
            node_map[name] = node
            if stack:
                parent = stack[-1]
                parent.children.append(node)
                node.parent = parent
            else:
                root = node
            stack.append(node)
        elif line.startswith("End Site"):
            i += 2
            offset = list(map(float, lines[i].split()[1:]))
            end = BVHNode(stack[-1].name + "_End", offset, [])
            stack[-1].children.append(end)
            i += 1
        elif line == "}":
            stack.pop()
        elif line.startswith("MOTION"):
            break
        i += 1

    # ---- 动作数据 ----
    while not lines[i].startswith("Frames:"):
        i += 1
    n_frames = int(lines[i].split(":")[1].strip())
    i += 1
    frame_time = float(lines[i].split(":")[1].strip())
    if not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH 帧间隔必须是有限正数")
    fps = 1.0 / frame_time

    motion = []
    for line in lines[i + 1:]:
        motion.append(list(map(float, line.split())))
    motion = np.asarray(motion, dtype=float)
    channel_count = sum(len(node.channels) for node in node_map.values())
    if root is None or n_frames < 1 or motion.shape != (n_frames, channel_count):
        raise ValueError("BVH 声明的帧数或通道数与动作数据不一致")
    if not np.isfinite(motion).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("BVH 动作数据和帧率必须是有限数值，帧率须为正数")
    return root, node_map, motion, fps


def forward_kinematics(root, motion):
    """计算每帧各节点的全局坐标"""
    joint_positions = {root.name: []}
    channel_index = 0

    def traverse(node, parent_pos, parent_rot, frame_data):
        nonlocal channel_index
        pos = node.offset.astype(float, copy=True)
        rot = R.identity()
        rotation_axes = []
        rotation_values = []
        channel_values = frame_data[channel_index:channel_index + len(node.channels)]
        for channel, value in zip(node.channels, channel_values):
            if channel.endswith("position"):
                pos["XYZ".index(channel[0])] += value
            elif channel.endswith("rotation"):
                rotation_axes.append(channel[0])
                rotation_values.append(value)
        if rotation_axes:
            angles = rotation_values[0] if len(rotation_values) == 1 else rotation_values
            rot = R.from_euler("".join(rotation_axes), angles, degrees=True)
        channel_index += len(node.channels)

        global_pos = parent_pos + parent_rot.apply(pos)
        global_rot = parent_rot * rot

        if node.name not in joint_positions:
            joint_positions[node.name] = []
        joint_positions[node.name].append(global_pos)

        for child in node.children:
            traverse(child, global_pos, global_rot, frame_data)

    # 对每帧执行 FK
    for frame in motion:
        channel_index = 0
        traverse(root, np.zeros(3), R.identity(), frame)

    # 转为 numpy
    for k in joint_positions:
        joint_positions[k] = np.array(joint_positions[k])

    return joint_positions


DEFAULT_JOINTS = ("LeftFoot", "LeftToe", "RightFoot", "RightToe")


def compute_contact_mask(
    bvh_file, *, joint_names=DEFAULT_JOINTS, up_axis="y", meters_per_unit=0.01,
    foot_height=0.10, toe_height=0.04, speed_threshold=0.20, output_path=None,
):
    """输出每帧的左右脚接触标记与用于判断的高度、速度。"""
    limits = np.asarray([meters_per_unit, foot_height, toe_height, speed_threshold])
    if (not np.isfinite(limits).all() or meters_per_unit <= 0 or
            min(foot_height, toe_height, speed_threshold) < 0):
        raise ValueError("单位换算系数须为有限正数，阈值须为有限非负数")
    if len(joint_names) != 4 or len(set(joint_names)) != 4:
        raise ValueError("请提供四个不同的脚与脚趾关节名")
    if up_axis not in ("x", "y", "z"):
        raise ValueError("竖直轴须为 x、y 或 z")

    source = Path(bvh_file)
    output = Path(output_path) if output_path else source.with_name(f"{source.stem}_contact_mask.json")
    if source.resolve() == output.resolve():
        raise ValueError("输出路径不能与输入相同")
    root, _, motion, fps = read_bvh(source)
    joints = forward_kinematics(root, motion)
    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise ValueError("BVH 找不到关节：" + "、".join(missing))

    positions = np.stack([joints[name] for name in joint_names], axis=1) * meters_per_unit
    heights = positions[:, :, "xyz".index(up_axis)].copy()
    frame_count = len(motion)
    velocities = np.gradient(positions, 1 / fps, axis=0) if frame_count > 1 else np.zeros_like(positions)
    speeds = np.linalg.norm(velocities, axis=-1)

    # 保留原处理流程中的首尾漂移、左右高度差和整体高度补偿。
    left_toe, right_toe = heights[:, 1], heights[:, 3]
    drift = min(left_toe[0] - left_toe[-1], right_toe[0] - right_toe[-1])
    heights += (drift * np.linspace(0.0, 1.0, frame_count))[:, None]
    heights[:, 2] += heights[:, 0].min() - heights[:, 2].min()
    heights[:, 3] += heights[:, 1].min() - heights[:, 3].min()
    min_toe = float(heights[:, [1, 3]].min())
    offset = -min_toe
    heights += offset

    height_limits = [foot_height, toe_height, foot_height, toe_height]
    below_limits = (heights < height_limits) & (speeds < speed_threshold)
    mask = below_limits.reshape(frame_count, 2, 2).all(axis=-1).astype(int)
    result = {
        "file": source.name,
        "fps": fps,
        "toe_origin_height": 0.0,
        "toe_min_height": min_toe,
        "offset_applied": offset,
        "foot_thresholds": {"height": foot_height, "speed": speed_threshold},
        "toe_thresholds": {"height": toe_height, "speed": speed_threshold},
        "contact_mask": mask.tolist(),
        "joint_data": {
            name: {"height": heights[:, index].tolist(), "speed": speeds[:, index].tolist()}
            for index, name in enumerate(joint_names)
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mask, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--joints", nargs=4, default=DEFAULT_JOINTS,
        metavar=("LEFT_FOOT", "LEFT_TOE", "RIGHT_FOOT", "RIGHT_TOE"),
    )
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--meters-per-unit", type=float, default=0.01)
    parser.add_argument("--foot-height", type=float, default=0.10, help="脚部高度阈值，单位 m")
    parser.add_argument("--toe-height", type=float, default=0.04, help="脚趾高度阈值，单位 m")
    parser.add_argument("--speed-threshold", type=float, default=0.20, help="速度阈值，单位 m/s")
    args = parser.parse_args()
    try:
        mask, fps = compute_contact_mask(
            args.input, joint_names=args.joints, up_axis=args.up_axis,
            meters_per_unit=args.meters_per_unit, foot_height=args.foot_height,
            toe_height=args.toe_height, speed_threshold=args.speed_threshold,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"已处理 {len(mask)} 帧，采样率 {fps:.2f} Hz")


if __name__ == "__main__":
    main()

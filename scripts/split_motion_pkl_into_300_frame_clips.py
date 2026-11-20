#!/usr/bin/env python3
"""
将动作数据集中的每个动作按固定帧数切片，并保存为新的 joblib PKL。

传入输入和输出 PKL 路径。默认每 300 帧切一段，丢弃不足一段的尾帧，
并用原动作名和 _clip 序号生成切片名称。

目标动作示例：
{
    "motion_clip0": {
        "root_trans_offset": ndarray [300, 3],
        "pose_aa":           ndarray [300, J, 3],
        "dof":               ndarray [300, D],
        "root_rot":          ndarray [300, 4],
        "smpl_joints":       ndarray [300, 24, 3],
        "fps":               int,
        "motion_name":       str,
    }
}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# 这些字段的第 0 维均表示时间帧，需要同步切片。
FRAME_FIELDS = (
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_rot",
    "smpl_joints",
)

REQUIRED_FIELDS = set(FRAME_FIELDS) | {"fps"}


def load_dataset(path: Path) -> dict[str, Any]:
    """读取 joblib PKL，并检查顶层结构。"""
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{path}")

    data = joblib.load(path)

    if not isinstance(data, dict):
        raise TypeError(
            f"输入文件顶层必须是 dict，实际为 {type(data)!r}"
        )

    return data


def validate_motion(
    motion_name: str,
    motion: dict[str, Any],
) -> int:
    """
    验证单条动作，并返回帧数 T。

    关节维数由输入数据决定，所有逐帧字段必须具有相同的帧数。
    """
    if not isinstance(motion, dict):
        raise TypeError(
            f"{motion_name}: 动作内容必须是 dict，"
            f"实际为 {type(motion)!r}"
        )

    missing_fields = REQUIRED_FIELDS.difference(motion)
    if missing_fields:
        raise KeyError(
            f"{motion_name}: 缺少字段 {sorted(missing_fields)}"
        )

    arrays: dict[str, np.ndarray] = {}

    for field in FRAME_FIELDS:
        value = np.asarray(motion[field])

        if value.ndim == 0:
            raise ValueError(
                f"{motion_name}.{field}: 不能是标量"
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{motion_name}.{field}: 存在 NaN 或 Inf"
            )

        arrays[field] = value

    frame_count = int(arrays["root_trans_offset"].shape[0])

    if arrays["root_trans_offset"].ndim != 2:
        raise ValueError(
            f"{motion_name}.root_trans_offset 应为二维数组，"
            f"实际 shape={arrays['root_trans_offset'].shape}"
        )

    if arrays["root_trans_offset"].shape[1] != 3:
        raise ValueError(
            f"{motion_name}.root_trans_offset 应为 [T, 3]，"
            f"实际 shape={arrays['root_trans_offset'].shape}"
        )

    if arrays["root_rot"].ndim != 2 or arrays["root_rot"].shape[1] != 4:
        raise ValueError(
            f"{motion_name}.root_rot 应为 [T, 4]，"
            f"实际 shape={arrays['root_rot'].shape}"
        )

    if (
        arrays["smpl_joints"].ndim != 3
        or arrays["smpl_joints"].shape[1:] != (24, 3)
    ):
        raise ValueError(
            f"{motion_name}.smpl_joints 应为 [T, 24, 3]，"
            f"实际 shape={arrays['smpl_joints'].shape}"
        )

    if arrays["dof"].ndim != 2:
        raise ValueError(
            f"{motion_name}.dof 应为二维数组 [T, D]，"
            f"实际 shape={arrays['dof'].shape}"
        )

    if arrays["pose_aa"].ndim != 3 or arrays["pose_aa"].shape[2] != 3:
        raise ValueError(
            f"{motion_name}.pose_aa 应为 [T, J, 3]，"
            f"实际 shape={arrays['pose_aa'].shape}"
        )

    for field, array in arrays.items():
        if array.shape[0] != frame_count:
            raise ValueError(
                f"{motion_name}: {field} 帧数为 {array.shape[0]}，"
                f"但 root_trans_offset 帧数为 {frame_count}"
            )

    try:
        fps = int(motion["fps"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{motion_name}.fps 无法转换为 int：{motion['fps']!r}"
        ) from exc

    if fps <= 0:
        raise ValueError(
            f"{motion_name}.fps 必须大于 0，实际为 {fps}"
        )

    return frame_count


def slice_motion(
    source_name: str,
    motion: dict[str, Any],
    *,
    clip_length: int,
    motion_name_mode: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    切分一条动作。

    返回：
    - 该动作产生的所有切片
    - 被丢弃的尾部帧数
    """
    frame_count = validate_motion(source_name, motion)
    clip_count = frame_count // clip_length
    discarded_frames = frame_count % clip_length

    clips: dict[str, dict[str, Any]] = {}

    for clip_index in range(clip_count):
        start = clip_index * clip_length
        end = start + clip_length
        clip_name = f"{source_name}_clip{clip_index}"

        if clip_name in clips:
            raise ValueError(f"生成了重复切片名称：{clip_name}")

        clip_motion: dict[str, Any] = {}

        # 逐帧字段同步切片，并复制为独立数组。
        for field in FRAME_FIELDS:
            source_array = np.asarray(motion[field])
            clip_motion[field] = np.array(
                source_array[start:end],
                copy=True,
            )

        # 标量/元数据字段。
        clip_motion["fps"] = int(motion["fps"])

        if motion_name_mode == "clip":
            clip_motion["motion_name"] = clip_name
        elif motion_name_mode == "source":
            clip_motion["motion_name"] = source_name
        else:
            raise ValueError(
                f"未知 motion_name_mode：{motion_name_mode}"
            )

        clips[clip_name] = clip_motion

    return clips, discarded_frames


def validate_clip(
    clip_name: str,
    clip: dict[str, Any],
    *,
    clip_length: int,
) -> None:
    """检查生成的单个切片结构和长度。"""
    frame_count = validate_motion(clip_name, clip)

    if frame_count != clip_length:
        raise ValueError(
            f"{clip_name}: 切片长度应为 {clip_length}，"
            f"实际为 {frame_count}"
        )

    if "motion_name" not in clip:
        raise KeyError(f"{clip_name}: 缺少 motion_name")

    if not isinstance(clip["motion_name"], str):
        raise TypeError(
            f"{clip_name}.motion_name 必须是 str，"
            f"实际为 {type(clip['motion_name'])!r}"
        )


def split_dataset(
    dataset: dict[str, Any],
    *,
    clip_length: int,
    motion_name_mode: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, int],
    list[str],
]:
    """
    切分整个数据集。

    返回：
    - 所有切片
    - 每个动作丢弃的尾部帧数
    - 长度不足一个完整切片、因此完全被舍弃的动作
    """
    output_data: dict[str, dict[str, Any]] = {}
    discarded_by_motion: dict[str, int] = {}
    fully_discarded_motions: list[str] = []

    total_motions = len(dataset)

    for motion_index, (source_name, motion) in enumerate(
        dataset.items(),
        start=1,
    ):
        clips, discarded_frames = slice_motion(
            source_name,
            motion,
            clip_length=clip_length,
            motion_name_mode=motion_name_mode,
        )

        if not clips:
            fully_discarded_motions.append(source_name)

        for clip_name, clip_motion in clips.items():
            if clip_name in output_data:
                raise ValueError(
                    f"不同动作生成了重复名称：{clip_name}"
                )

            validate_clip(
                clip_name,
                clip_motion,
                clip_length=clip_length,
            )
            output_data[clip_name] = clip_motion

        discarded_by_motion[source_name] = discarded_frames

        original_frames = (
            np.asarray(motion["root_trans_offset"]).shape[0]
        )

        print(
            f"[{motion_index:03d}/{total_motions:03d}] "
            f"{source_name}: "
            f"原始帧数={original_frames}, "
            f"生成切片={len(clips)}, "
            f"丢弃尾帧={discarded_frames}"
        )

    return (
        output_data,
        discarded_by_motion,
        fully_discarded_motions,
    )


def inspect_saved_file(
    output_path: Path,
    *,
    preview_count: int,
) -> None:
    """重新读取保存结果并打印前几个切片。"""
    saved_data = joblib.load(output_path)

    print("\n重新读取输出文件进行验证")
    print("动作总数：", len(saved_data))
    print("顶层类型：", type(saved_data))

    for clip_name, clip in list(saved_data.items())[:preview_count]:
        print(f"\n动作名称：{clip_name}")
        print("值类型：", type(clip))

        for field, value in clip.items():
            print(
                "  ",
                field,
                "| 类型：",
                type(value),
                "| shape：",
                getattr(value, "shape", None),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将动作数据按固定长度连续切片，"
            "不足一个完整切片的尾部直接舍弃"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 PKL 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 PKL 路径",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=300,
        help="每个切片的帧数，默认 300",
    )
    parser.add_argument(
        "--motion-name-mode",
        choices=("clip", "source"),
        default="clip",
        help=(
            "motion_name 的内容："
            "clip=切片名称，source=原动作名称；默认 clip"
        ),
    )
    parser.add_argument(
        "--compress",
        type=int,
        choices=range(0, 10),
        default=0,
        metavar="0-9",
        help="joblib 压缩等级，0 表示不压缩，默认 0",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help="保存后打印前几个切片，默认 3",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clip_length <= 0:
        raise ValueError(
            f"clip-length 必须大于 0，实际为 {args.clip_length}"
        )

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if input_path == output_path:
        raise ValueError(
            "输出路径不能和输入路径相同，避免覆盖原始数据"
        )

    print("输入文件：", input_path)
    print("输出文件：", output_path)
    print("切片长度：", args.clip_length)
    print("motion_name 模式：", args.motion_name_mode)

    dataset = load_dataset(input_path)

    print("\n原始动作总数：", len(dataset))

    (
        output_data,
        discarded_by_motion,
        fully_discarded_motions,
    ) = split_dataset(
        dataset,
        clip_length=args.clip_length,
        motion_name_mode=args.motion_name_mode,
    )

    if not output_data:
        raise RuntimeError(
            "没有生成任何完整切片，未写入输出文件"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        output_data,
        output_path,
        compress=args.compress,
    )

    total_discarded_frames = sum(
        discarded_by_motion.values()
    )

    print("\n切片完成")
    print("原始动作数：", len(dataset))
    print("生成切片数：", len(output_data))
    print("总丢弃尾帧数：", total_discarded_frames)
    print(
        "完全不足一个切片而被舍弃的动作数：",
        len(fully_discarded_motions),
    )

    if fully_discarded_motions:
        print("完全舍弃的动作：")
        for motion_name in fully_discarded_motions:
            print(f"  - {motion_name}")

    print("保存位置：", output_path)
    print(
        "文件大小："
        f"{output_path.stat().st_size / 1024**2:.2f} MiB"
    )

    inspect_saved_file(
        output_path,
        preview_count=args.preview_count,
    )


if __name__ == "__main__":
    main()

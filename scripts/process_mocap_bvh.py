#!/usr/bin/env python3
"""按时间轴对 BVH 降采样，同时更新帧数和帧间隔。"""

import argparse
import math
from pathlib import Path

from downsample_processed_motion import downsample_frame_indices


def downsample_bvh(input_path, fps_target, str_target=None, output_path=None):
    """保留 BVH 层级，按最接近目标采样时间的输入帧重建动作段。"""
    source = Path(input_path)
    default_name = f"{source.stem}_{fps_target:g}fps{source.suffix}"
    output = Path(output_path) if output_path else source.with_name(default_name)
    if source.resolve() == output.resolve():
        raise ValueError("输出路径不能与输入相同")
    if not math.isfinite(fps_target) or fps_target <= 0:
        raise ValueError("目标帧率必须是有限正数")

    lines = source.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    if str_target:
        lines = [line.replace(str_target, "") for line in lines]

    for index, line in enumerate(lines):
        if line.strip().startswith("Frame Time:"):
            break
    else:
        raise ValueError("BVH 缺少 Frame Time 字段")
    if index == 0 or not lines[index - 1].strip().startswith("Frames:"):
        raise ValueError("BVH 缺少 Frames 字段")

    frame_time = float(line.split(":", 1)[1])
    if not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("Frame Time 必须是有限正数")
    fps_source = 1.0 / frame_time
    ratio = fps_source / fps_target
    stride = round(ratio)
    if stride >= 1 and math.isclose(ratio, stride, rel_tol=1e-4):
        # 容许 BVH 帧间隔的舍入误差，例如 120 Hz 写作 0.008333 s。
        fps_source = stride * fps_target

    motion = [row for row in lines[index + 1:] if row.strip()]
    indices = downsample_frame_indices(len(motion), fps_source, fps_target)
    header = lines[:index - 1] + [
        f"Frames: {len(indices)}\n",
        f"Frame Time: {1.0 / fps_target:.9f}\n",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(header + [motion[i] for i in indices]), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="输入 BVH 文件")
    parser.add_argument("--output", type=Path, help="输出 BVH 文件，默认与输入同目录")
    parser.add_argument("--fps_target", required=True, type=float, help="目标帧率")
    parser.add_argument("--remove_str", help="从 BVH 中移除的字串")
    args = parser.parse_args()
    try:
        output = downsample_bvh(args.input, args.fps_target, args.remove_str, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"输出文件：{output}")


if __name__ == "__main__":
    main()

# 机器人动作数据处理

这套工具用于人形机器人动作数据的格式转换、降采样、足部接触检测和曲线编辑。输入动作、关节映射和机器人描述由使用者提供。

| 功能 | 入口 |
| --- | --- |
| GMR NPZ 与编辑器 NPZ 双向转换 | `npz_grm_to_robot_edited_format.py`、`npz_robot_edited_to_gmr_format.py` |
| 双向批量转换 | `npz_grm_to_robot_edited_format_batch.py`、`npz_robot_edited_to_gmr_format_batch.py` |
| SOMA CSV 转编辑器 NPZ | `soma_csv_to_robot_edited_npz_batch.py` |
| NPZ / PKL 动作同步降采样 | `downsample_processed_motion.py` |
| BVH 降采样 | `process_mocap_bvh.py` |
| BVH 足部接触检测 | `cal_contact_mask.py` |
| 动作数据集定长切片 | `split_motion_pkl_into_300_frame_clips.py` |
| 按脚踝高度纠正根节点高度 | `auto_ground_editor_npz_batch.py` |
| PKL 与 NPZ 互转 | `pkl_to_npz.py`、`npz_to_pkl.py` |
| Web 动作编辑、局部平滑、裁剪和 NPZ 导出 | `editor/app.py` |

## 使用

```sh
python -m pip install -r requirements.txt

python scripts/npz_grm_to_robot_edited_format.py \
  --input data/motion.npz --output outputs/edited.npz \
  --quat_order xyzw --joint_names_file data/joints.txt

python scripts/downsample_processed_motion.py \
  --input outputs/edited.npz --output outputs/edited_30hz.npz \
  --source_fps 120 --target_fps 30

python scripts/process_mocap_bvh.py \
  --input data/motion.bvh --output outputs/motion_30hz.bvh --fps_target 30

python scripts/cal_contact_mask.py \
  --input outputs/motion_30hz.bvh --output outputs/contact.json \
  --joints LeftFoot LeftToe RightFoot RightToe --up-axis y --meters-per-unit 0.01

python scripts/auto_ground_editor_npz_batch.py \
  --src_folder data/editor_motions --tgt_folder outputs/grounded \
  --urdf data/robot.urdf --left_ankle_joint ankle_left \
  --right_ankle_joint ankle_right --sole_offset_m 0.05
```

GMR 转换所用的关节名必须与动作列顺序一致，可来自输入的 `joint_names`、关节名文件或外部 URDF。贴地脚本以两侧脚踝原点的较低高度估算支撑侧，根据传入的脚底偏移修正根节点 Z 坐标。`--sole_offset_m` 应按使用的模型填写，示例值仅作参数演示。

每个脚本支持 `--help`。PKL 和带对象数组的 NPY / NPZ 请使用可信来源。

## Web 编辑器与合成演示

从项目根目录运行 `python editor/app.py`，浏览器访问 `http://127.0.0.1:5000`。编辑器使用 CDN 加载 Three.js 和 URDFLoader。仅查看和编辑曲线时，直接导入 NPZ 即可。下面的示例生成两条合成关节曲线，可用来演示选区平滑、裁剪和导出。

```python
from pathlib import Path

import numpy as np

demo = Path("outputs/synthetic_motion.npz")
demo.parent.mkdir(exist_ok=True)
fps = 30
t = np.arange(120) / fps
np.savez(
    demo,
    joint_names=np.array(["demo_joint_1", "demo_joint_2"]),
    joint_pos=np.column_stack([
        0.4 * np.sin(2 * np.pi * t) + 0.03 * np.sin(18 * np.pi * t),
        0.3 * np.cos(2 * np.pi * t),
    ]),
    base_pos_w=np.zeros((len(t), 3)),
    base_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1)),
    framerate=np.array(fps),
)
```

3D 回放时，可把 `MOTION_EDITOR_MODEL_DIR` 设为公开模型所在目录的绝对路径，再启动编辑器。也可以把模型和网格放到已被 Git 忽略的 `editor/static/` 中。保留 URDF 到网格的相对引用，在界面中填写 URDF 文件名。用于回放的 NPZ 关节名需对应模型。贴地功能需要在界面填写该模型的足部关节、连杆或脚底偏移。

公开模型可参考 [NVIDIA SOMA Retargeter 的机器人资产许可](https://github.com/NVIDIA/soma-retargeter#license)。

## 输入格式

- [GMR](https://github.com/YanjieZe/GMR)
- [NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)

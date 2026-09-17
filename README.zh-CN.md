# AT-YOLO11

[English](README.md)

AT-YOLO11 是面向清晰、中度浑浊和严重浑浊水下环境的鱼类目标检测模型。本仓库包含最终模型实现、模型和数据集配置、浑浊度标定文件，以及可复现的训练入口。

## 方法

默认模型包含：

- 用于提取图像级浑浊度特征的深层 `TurbidityHead`；
- 用于调制 neck P4 特征的 `TABlock`；
- 三尺度 YOLO 检测头。

默认模型配置为 [`ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml`](ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml)。

## 数据集

数据集采用 YOLO 格式，类别为 `fish`：

| 目录 | 水体条件 | 训练集 | 验证集 | 测试集 |
| --- | --- | ---: | ---: | ---: |
| `expqingxi` | 清晰 | 1,200 | 200 | 200 |
| `expzhonghun` | 中度浑浊 | 1,200 | 200 | 200 |
| `expyz` | 严重浑浊 | 1,200 | 200 | 200 |

从 [`dataset-v1.0` Release](https://github.com/liuxiaodada/AT-YOLO11/releases/tag/dataset-v1.0) 下载 `expqingxi.tar.gz`、`expzhonghun.tar.gz` 和 `expyz.tar.gz`，然后解压到 `datasets/`：

```bash
mkdir -p datasets
tar -xzf expqingxi.tar.gz -C datasets
tar -xzf expzhonghun.tar.gz -C datasets
tar -xzf expyz.tar.gz -C datasets
```

解压后的目录结构应为：

```text
datasets/
├── expqingxi/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── expzhonghun/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── expyz/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

## 安装

需要 Python 3.8 或更高版本。

```bash
git clone https://github.com/liuxiaodada/AT-YOLO11.git
cd AT-YOLO11
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 训练与测试

仓库中的 `turbidity_pseudo_calibration.json` 根据发布的训练集计算。如果替换或修改训练数据，需要重新标定：

```bash
python tools/calibrate_turbidity_pseudo.py
```

先运行单个随机种子进行检查：

```bash
python tools/run_at_yolov11_seeds.py --seeds 0 --epochs 150 --batch 32 --imgsz 640 --device 0
```

运行默认的五随机种子实验：

```bash
python tools/run_at_yolov11_seeds.py --seeds 0 1 2 3 4 --epochs 150 --batch 32 --imgsz 640 --device 0
```

该脚本使用 `expyz/images/train` 训练，在三个浑浊度验证集上验证，并分别评估三个独立测试集。结果保存在 `runs/detect/at_yolov11_seed_experiments/`。

## 主要文件

- `ultralytics/data/turbidity.py`：浑浊度特征与伪标签工具。
- `ultralytics/data/dataset.py`：数据增强前的浑浊度监督目标集成。
- `ultralytics/nn/modules/conv.py`：`TurbidityHead` 和 `TABlock` 实现。
- `ultralytics/nn/tasks.py`：AT-YOLO11 前向传播和模型解析。
- `ultralytics/utils/loss.py`：AT-YOLO11 训练损失集成。
- `ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml`：最终模型结构。
- `tools/run_at_yolov11_seeds.py`：多随机种子训练及三种水体条件测试。
- `tools/calibrate_turbidity_pseudo.py`：基于训练集的伪浑浊度校准。

## 许可协议

代码基于 Ultralytics，并按照 [AGPL-3.0 License](LICENSE) 发布。数据集使用和再分发条款将随数据集 Release 一同提供。

## 致谢

本项目基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 开发。

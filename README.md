# AT-YOLO11

[中文说明](README.zh-CN.md)

AT-YOLO11 is an underwater fish detector designed for clear, moderate-turbidity, and severe-turbidity conditions. This repository contains the final model implementation, model and dataset configurations, turbidity calibration, and the reproducible training entry point.

## Method

The default model configuration combines:

- a deep `TurbidityHead` that estimates image-level turbidity features;
- a `TABlock` that modulates the P4 neck feature;
- three-scale YOLO detection heads.

The default configuration is [`ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml`](ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml).

## Datasets

The dataset release contains three YOLO-format subsets with one class, `fish`:

| Directory | Condition | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `expqingxi` | Clear | 1,200 | 200 | 200 |
| `expzhonghun` | Moderate turbidity | 1,200 | 200 | 200 |
| `expyz` | Severe turbidity | 1,200 | 200 | 200 |

Download `expqingxi.tar.gz`, `expzhonghun.tar.gz`, and `expyz.tar.gz` from the [`dataset-v1.0` release](https://github.com/liuxiaodada/AT-YOLO11/releases/tag/dataset-v1.0), then extract them into `datasets/`:

```bash
mkdir -p datasets
tar -xzf expqingxi.tar.gz -C datasets
tar -xzf expzhonghun.tar.gz -C datasets
tar -xzf expyz.tar.gz -C datasets
```

The resulting layout should be:

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

## Installation

Python 3.8 or later is required.

```bash
git clone https://github.com/liuxiaodada/AT-YOLO11.git
cd AT-YOLO11
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training and evaluation

The included `turbidity_pseudo_calibration.json` was calculated from the released training split. Recalculate it after replacing or changing the training set:

```bash
python tools/calibrate_turbidity_pseudo.py
```

Run one seed as a quick check:

```bash
python tools/run_at_yolov11_seeds.py --seeds 0 --epochs 150 --batch 32 --imgsz 640 --device 0
```

Run the default five-seed experiment:

```bash
python tools/run_at_yolov11_seeds.py --seeds 0 1 2 3 4 --epochs 150 --batch 32 --imgsz 640 --device 0
```

The script trains with `expyz/images/train`, validates across all three turbidity levels, and evaluates each held-out test set. Outputs are written under `runs/detect/at_yolov11_seed_experiments/`.

## Project files

- `ultralytics/data/turbidity.py`: turbidity feature and pseudo-label utilities.
- `ultralytics/data/dataset.py`: pre-augmentation turbidity target integration.
- `ultralytics/nn/modules/conv.py`: `TurbidityHead` and `TABlock` implementations.
- `ultralytics/nn/tasks.py`: AT-YOLO11 forward routing and model parsing.
- `ultralytics/utils/loss.py`: AT-YOLO11 training loss integration.
- `ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml`: final model architecture.
- `tools/run_at_yolov11_seeds.py`: repeatable multi-seed training and three-condition evaluation.
- `tools/calibrate_turbidity_pseudo.py`: training-set pseudo-turbidity calibration.

## License

The code is based on Ultralytics and is distributed under the [AGPL-3.0 License](LICENSE). Dataset terms are provided with the dataset release.

## Acknowledgments

This project builds on [Ultralytics](https://github.com/ultralytics/ultralytics).

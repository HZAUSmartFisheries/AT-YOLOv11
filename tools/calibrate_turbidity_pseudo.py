"""Calibrate AT-YOLO11 turbidity pseudo-label normalization on the training set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse calibration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "yolo_dataset_combined_val.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=ROOT / "turbidity_pseudo_calibration.json")
    return parser.parse_args()


def main() -> None:
    """Calculate and save fixed log-space quantiles using training images only."""
    args = parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    os.environ["TURBIDITY_PSEUDO_SOURCE"] = "preaugment"
    cfg = get_cfg(
        overrides={
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": 0,
            "task": "detect",
            "mode": "train",
            "rect": False,
            "cache": False,
            "single_cls": False,
            "fraction": 1.0,
            "mosaic": 0.0,
        }
    )
    data = check_det_dataset(str(args.data.resolve()), autodownload=False)
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", stride=32)
    log_values = []
    for start in range(0, len(dataset), args.batch):
        stop = min(start + args.batch, len(dataset))
        raw_values = torch.stack([dataset.get_image_and_label(index)["raw_t_preaugment"] for index in range(start, stop)])
        log_values.append(torch.log1p(raw_values).numpy())
        print(f"Processed {stop}/{len(dataset)}", end="\r", flush=True)
    values = np.concatenate(log_values)
    q_low, q_high = np.quantile(values, [args.lower_quantile, args.upper_quantile])
    calibration = {
        "data": str(args.data.resolve()),
        "split": "train",
        "imgsz": args.imgsz,
        "source": "preaugment",
        "n": len(values),
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "log_raw_q05": float(q_low),
        "log_raw_q95": float(q_high),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, args.output)
    normalized = np.clip((values - q_low) / (q_high - q_low + 1e-6), 0, 1)
    print(f"\nSaved calibration to: {args.output.resolve()}")
    print(json.dumps(calibration, indent=2, ensure_ascii=False))
    print(
        f"Normalized train distribution: mean={normalized.mean():.6f}, std={normalized.std(ddof=1):.6f}, "
        f"range=[{normalized.min():.6f}, {normalized.max():.6f}]"
    )


if __name__ == "__main__":
    main()

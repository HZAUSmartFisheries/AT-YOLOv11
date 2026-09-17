"""Train and evaluate AT-YOLO11 across reproducible random seeds."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/11/yolo11-TurbidityHead-Deep-TABlock-P4.yaml"
DEFAULT_PRETRAINED = ROOT / "yolo11n.pt"
DEFAULT_DATA = ROOT / "yolo_dataset_combined_val.yaml"
DEFAULT_PROJECT = ROOT / "runs/detect/at_yolov11_seed_experiments"
DEFAULT_RUN_PREFIX = "at_yolo11"
TEST_DATASETS = {
    "clear": ROOT / "yolo_datasetqingxi.yaml",
    "moderate": ROOT / "yolo_datasetzhonghun.yaml",
    "severe": ROOT / "yolo_bvn.yaml",
}
METRIC_FIELDS = ("model", "seed", "turbidity", "precision", "recall", "mAP50", "mAP50_95", "weights")


def parse_args() -> argparse.Namespace:
    """Parse configurable model, training, and output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="AT-YOLO11", help="Model label written to logs and test_metrics.csv.")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate. Omit to keep optimizer=auto; providing a value uses explicit AdamW.",
    )
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate as a fraction of lr0.")
    parser.add_argument("--momentum", type=float, default=0.9, help="AdamW beta1 used with an explicit lr0.")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing seed directory. Without this flag, existing runs are skipped.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without training.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and test existing best.pt files.")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the three held-out test sets.")
    args = parser.parse_args()
    for field in ("model", "pretrained", "data", "project"):
        path = getattr(args, field).expanduser()
        setattr(args, field, path if path.is_absolute() else (ROOT / path).resolve())
    return args


def validate_paths(args: argparse.Namespace) -> None:
    """Fail early when a required model, checkpoint, or dataset file is missing."""
    required = [("model", args.model), ("pretrained", args.pretrained), ("data", args.data)]
    if not args.skip_test:
        required.extend((f"{name} test data", path) for name, path in TEST_DATASETS.items())
    for label, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read previously completed test metrics."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write test metrics after every completed evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    """Train the same AT-YOLOv11 configuration once for each requested seed."""
    args = parse_args()
    validate_paths(args)

    import ultralytics
    from ultralytics import YOLO

    imported_package = Path(ultralytics.__file__).resolve()
    if ROOT not in imported_package.parents:
        raise RuntimeError(f"Wrong Ultralytics package imported: {imported_package}; expected it under {ROOT}")

    print(f"Using Ultralytics package: {imported_package}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    metrics_path = args.project / "test_metrics.csv"
    metric_rows = read_metrics(metrics_path)

    for seed in args.seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        run_dir = args.project / run_name
        weights = run_dir / "weights/best.pt"
        completion_marker = run_dir / "TRAINING_COMPLETED.txt"

        optimizer_args = (
            {"optimizer": "auto"}
            if args.lr0 is None
            else {
                "optimizer": "AdamW",
                "lr0": args.lr0,
                "lrf": args.lrf,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            }
        )

        train_args = {
            "data": str(args.data),
            "epochs": args.epochs,
            "workers": args.workers,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "project": str(args.project),
            "name": run_name,
            "exist_ok": True,
            "seed": seed,
            "deterministic": True,
            "patience": args.patience,
            "mosaic": args.mosaic,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            **optimizer_args,
        }

        print(f"\n=== {args.model_name} seed={seed} ===")
        print(f"Output: {run_dir}")
        if args.dry_run:
            print(train_args)
            if not args.skip_test:
                print({"test": {name: str(path) for name, path in TEST_DATASETS.items()}})
            continue

        if args.overwrite:
            metric_rows = [row for row in metric_rows if int(row["seed"]) != seed]

        if not args.eval_only and (args.overwrite or not weights.is_file()):
            model = YOLO(str(args.model))
            model.load(str(args.pretrained))
            model.train(**train_args)
            if not weights.is_file():
                raise FileNotFoundError(f"Training finished without producing: {weights}")
            completion_marker.write_text(
                f"model={args.model_name}\nseed={seed}\nepochs={args.epochs}\nweights={weights}\n",
                encoding="utf-8",
            )
        elif weights.is_file():
            print(f"Skipping completed training and using existing weights: {weights}")

        if not weights.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {weights}")

        if args.skip_test:
            continue

        completed = {(int(row["seed"]), row["turbidity"]) for row in metric_rows}
        for turbidity, test_data in TEST_DATASETS.items():
            if (seed, turbidity) in completed:
                print(f"Skipping completed test: seed={seed}, turbidity={turbidity}")
                continue

            metrics = YOLO(str(weights)).val(
                data=str(test_data),
                split="test",
                workers=args.workers,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                project=str(args.project / "test_runs"),
                name=f"{run_name}_{turbidity}",
                exist_ok=True,
                plots=False,
            )
            metric_rows.append(
                {
                    "model": args.model_name,
                    "seed": seed,
                    "turbidity": turbidity,
                    "precision": metrics.box.mp,
                    "recall": metrics.box.mr,
                    "mAP50": metrics.box.map50,
                    "mAP50_95": metrics.box.map,
                    "weights": weights,
                }
            )
            write_metrics(metrics_path, metric_rows)
            print(f"Saved test result: seed={seed}, turbidity={turbidity}, mAP50={metrics.box.map50:.6f}")


if __name__ == "__main__":
    main()

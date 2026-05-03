"""Custom YOLOv8 training script for SIHA FPV.

Trains a YOLOv8 model on the dataset defined in ``data/fpv.yaml``.
Results (weights, metrics, confusion matrix) are saved to ``runs/train/``.

Usage::

    # Fine-tune YOLOv8n on the FPV dataset (CPU, 50 epochs)
    python scripts/train.py

    # Use a larger model on GPU for higher accuracy
    python scripts/train.py --model yolov8s.pt --device cuda:0 --epochs 100

    # Resume an interrupted run
    python scripts/train.py --resume runs/train/exp/weights/last.pt

    # Export best weights to ONNX after training
    python scripts/train.py --export onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root (one level up from scripts/)
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="siha-train",
        description="SIHA – YOLOv8 custom model training",
    )
    p.add_argument(
        "--data",   default=str(ROOT / "data" / "fpv.yaml"),
        help="Path to dataset YAML (default: data/fpv.yaml)",
    )
    p.add_argument(
        "--model",  default="yolov8n.pt",
        help="Base weights to fine-tune (default: yolov8n.pt)",
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)",
    )
    p.add_argument(
        "--imgsz",  type=int, default=640,
        help="Input image size (default: 640)",
    )
    p.add_argument(
        "--batch",  type=int, default=16,
        help="Batch size; use -1 for auto (default: 16)",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device: cpu, cuda:0, mps (default: cpu)",
    )
    p.add_argument(
        "--project", default=str(ROOT / "runs" / "train"),
        help="Output directory (default: runs/train)",
    )
    p.add_argument(
        "--name",   default="siha_fpv",
        help="Run name (default: siha_fpv)",
    )
    p.add_argument(
        "--resume", metavar="WEIGHTS",
        help="Resume training from a checkpoint (.pt file)",
    )
    p.add_argument(
        "--export", metavar="FORMAT",
        help="Export best weights after training (onnx, tflite, engine, …)",
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience in epochs (default: 20)",
    )
    return p.parse_args()


def train(args: argparse.Namespace) -> Path:
    """Run YOLOv8 training and return path to best weights."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics")
        sys.exit(1)

    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        logger.info(
            "Training %s on %s for %d epochs (device=%s)",
            args.model, args.data, args.epochs, args.device,
        )
        model = YOLO(args.model)
        results = model.train(
            data      = args.data,
            epochs    = args.epochs,
            imgsz     = args.imgsz,
            batch     = args.batch,
            device    = args.device,
            project   = args.project,
            name      = args.name,
            patience  = args.patience,
            exist_ok  = True,
            verbose   = True,
        )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)
    return best_weights


def export(weights_path: Path, fmt: str) -> None:
    """Export trained weights to the specified format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    logger.info("Exporting %s → %s", weights_path, fmt)
    model = YOLO(str(weights_path))
    model.export(format=fmt)
    logger.info("Export complete")


def validate(weights_path: Path, data: str) -> None:
    """Run validation on the best weights and print metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    model = YOLO(str(weights_path))
    metrics = model.val(data=data)
    logger.info(
        "Validation  mAP50=%.3f  mAP50-95=%.3f",
        metrics.box.map50, metrics.box.map,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    best = train(args)

    validate(best, args.data)

    if args.export:
        export(best, args.export)


if __name__ == "__main__":
    main()

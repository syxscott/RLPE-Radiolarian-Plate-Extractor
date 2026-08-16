#!/usr/bin/env python3
"""Fine-tune a single-class YOLOv8 detector on RLPE panel crops."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "yolo_dataset" / "data.yaml"
DEFAULT_MODEL = REPO_ROOT / "models" / "radiolarian_yolo_v1.pt"
DEFAULT_RUNS = REPO_ROOT / "work" / "yolo_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_path = args.output.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO data configuration not found: {data_path}")

    run_name = output_path.stem
    started = time.perf_counter()
    model = YOLO(args.pretrained)
    metrics = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(DEFAULT_RUNS),
        name=run_name,
        exist_ok=True,
    )

    best_path = Path(metrics.save_dir) / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce best.pt at {best_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output_path)

    trained = YOLO(str(output_path))
    validation = trained.val(data=str(data_path), imgsz=args.imgsz, device=args.device)
    export_path = Path(
        trained.export(format="onnx", imgsz=args.imgsz, device=args.device, simplify=False)
    )
    target_onnx = output_path.with_suffix(".onnx")
    if export_path.resolve() != target_onnx.resolve():
        shutil.move(str(export_path), target_onnx)

    elapsed = time.perf_counter() - started
    print("\nTraining complete")
    print(f"Training time: {elapsed:.1f} seconds")
    print(f"Model classes: {trained.names}")
    print(f"mAP50: {validation.box.map50:.6f}")
    print(f"mAP50-95: {validation.box.map:.6f}")
    print(f"PyTorch model: {output_path} ({output_path.stat().st_size} bytes)")
    print(f"ONNX model: {target_onnx} ({target_onnx.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

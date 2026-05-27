from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import cv2
import numpy as np

if sys.platform != "win32":
    pathlib.WindowsPath = Path
    sys.modules["pathlib"].WindowsPath = Path


def _find_model() -> Path:
    root = Path(__file__).resolve().parent.parent
    for base in [Path.cwd(), Path.cwd() / "vision", root]:
        pt = base / "data" / "checkpoints" / "best" / "best_20260110_122310.pt"
        if pt.exists():
            return pt
        pt = base / "data" / "checkpoints" / "best" / "best.pt"
        if pt.exists():
            return pt
    raise FileNotFoundError("No best*.pt in data/checkpoints/best")


def _find_benchmark_image() -> Path:
    root = Path(__file__).resolve().parent.parent
    name = "Modern_Fianchetto_Setup._Chess_game_Staunton_No._6.jpg"
    for base in [Path.cwd(), Path.cwd() / "vision", root]:
        for p in [base / "data" / name, base / name]:
            if p.exists():
                return p
    raise FileNotFoundError(f"Benchmark image not found: {name}")


def run(
    model_path: Path | str | None = None,
    image_path: Path | str | None = None,
    output_path: Path | str | None = None,
    conf: float = 0.25,
) -> None:
    from ultralytics import YOLO

    model_path = Path(model_path) if model_path else _find_model()
    image_path = Path(image_path) if image_path else _find_benchmark_image()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    model = YOLO(str(model_path))
    print("model.names:", model.names)
    print("num classes:", len(model.names))

    results = model.predict(
        img,
        conf=conf,
        iou=0.45,
        verbose=False,
        imgsz=640,
    )
    r0 = results[0]
    boxes = r0.boxes
    n = len(boxes) if boxes is not None else 0
    print(f"detections (conf>={conf}): {n}")

    if n == 0 and conf >= 0.1:
        low = model.predict(img, conf=0.01, iou=0.45, verbose=False, imgsz=640)
        nlow = len(low[0].boxes) if low[0].boxes is not None else 0
        print(f"detections (conf>=0.01): {nlow}")
        if nlow > 0:
            print("Model produces low-confidence detections; try --conf 0.01 or check threshold.")

    out = img.copy()
    if boxes is not None and n > 0:
        for i in range(n):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            c = int(boxes.cls[i])
            cf = float(boxes.conf[i])
            try:
                name = model.names[c] if isinstance(model.names, (list, tuple)) else model.names.get(c, f"class_{c}")
            except (KeyError, TypeError, IndexError):
                name = f"class_{c}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {cf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), out)
        print(f"output: {p}")
    else:
        default = Path(image_path).parent / "benchmark_annotated.jpg"
        cv2.imwrite(str(default), out)
        print(f"output: {default}")

    if n == 0:
        print("0 detections on benchmark. If OAK also gives 0, likely: preprocessing (BGR/RGB, scale) or decode layout.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run chess YOLO on benchmark image to verify weights")
    ap.add_argument("--model", default=None, help="Path to .pt (default: data/checkpoints/best/best_*.pt)")
    ap.add_argument("--image", default=None, help="Path to image (default: data/Modern_Fianchetto_Setup.*.jpg)")
    ap.add_argument("--output", default=None, help="Path to save annotated image")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = ap.parse_args()
    run(model_path=args.model, image_path=args.image, output_path=args.output, conf=args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

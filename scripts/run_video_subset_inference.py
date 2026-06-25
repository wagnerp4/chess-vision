import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from scripts.evaluate_models import (
    IMG_EXTENSIONS,
    load_yolo_boxes,
    resolve_dataset_yaml,
    split_images_dir,
    yolo_labels_dir,
)
from src.data.class_mapping import CANONICAL_CLASSES
from src.live_pipeline import default_model
from src.metrics import evaluate_model_predictions

DEFAULT_DATA = VISION_ROOT / "data" / "processed" / "neuroTUM-chess-dataset" / "dataset.yaml"
DEFAULT_OUTPUT = VISION_ROOT / "results" / "video_subset_inference"


def resolve_checkpoint(model_arg: str | None) -> Path:
    if model_arg:
        path = Path(model_arg)
        if not path.is_absolute():
            path = VISION_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        return path
    preferred = VISION_ROOT / "data" / "checkpoints" / "best" / "best_20260527_yolo26n_chess_pieces_2.pt"
    if preferred.is_file():
        return preferred
    return default_model()


def list_test_images(yaml_path: Path) -> List[Path]:
    images_dir = split_images_dir(yaml_path, "test")
    paths = [
        p for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images under {images_dir}")
    return paths


def yolo_result_to_predictions(result, model) -> List[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = model.names
    out: List[dict] = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        if isinstance(names, dict):
            name = names.get(cls_id, f"class_{cls_id}")
        else:
            name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
        out.append({
            "class": name,
            "bbox": xyxy,
            "confidence": float(boxes.conf[i]),
        })
    return out


def draw_yolo_predictions(image: np.ndarray, predictions: List[dict]) -> np.ndarray:
    out = image.copy()
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        label = f"{pred['class']} {pred['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1, lh + 6)
        cv2.rectangle(out, (x1, ly - lh - 6), (x1 + lw, ly), (0, 255, 0), -1)
        cv2.putText(out, label, (x1, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO checkpoint inference on a shuffled subset of video pseudo-label frames"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to dataset.yaml or processed dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO .pt weights (default: best archived checkpoint)",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Number of frames to infer on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for optional metrics")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Directory for annotated images and results.json",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compare predictions vs pseudo-GT labels on the sampled frames",
    )
    args = parser.parse_args()

    yaml_path = resolve_dataset_yaml(args.data)
    model_path = resolve_checkpoint(args.model)
    all_images = list_test_images(yaml_path)
    sample_size = min(args.sample_size, len(all_images))

    rng = random.Random(args.seed)
    shuffled = list(all_images)
    rng.shuffle(shuffled)
    sample_paths = shuffled[:sample_size]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = VISION_ROOT / output_dir
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    id_to_name = {i: n for i, n in enumerate(CANONICAL_CLASSES)}
    labels_dir = yolo_labels_dir(split_images_dir(yaml_path, "test"))

    per_image: List[dict] = []
    predictions_map: Dict[str, List[dict]] = {}
    ground_truth_map: Dict[str, List[dict]] = {}
    t0 = time.perf_counter()

    for img_path in sample_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        h, w = img.shape[:2]
        t_img = time.perf_counter()
        results = model.predict(img, conf=args.conf, iou=0.45, verbose=False)
        infer_ms = round((time.perf_counter() - t_img) * 1000.0, 2)
        preds = yolo_result_to_predictions(results[0], model)
        stem = img_path.stem
        predictions_map[stem] = preds
        annotated = draw_yolo_predictions(img, preds)
        out_path = annotated_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        gt_path = labels_dir / f"{stem}.txt"
        ground_truth_map[stem] = load_yolo_boxes(gt_path, w, h, id_to_name)
        per_image.append({
            "stem": stem,
            "image": str(img_path.name),
            "num_detections": len(preds),
            "num_pseudo_gt": len(ground_truth_map[stem]),
            "infer_ms": infer_ms,
            "annotated": str(out_path.relative_to(output_dir)),
        })

    elapsed = round(time.perf_counter() - t0, 2)
    results_payload: dict = {
        "dataset_yaml": str(yaml_path.resolve()),
        "model": str(model_path.resolve()),
        "sample_size": sample_size,
        "total_frames": len(all_images),
        "seed": args.seed,
        "conf": args.conf,
        "iou": args.iou,
        "shuffled_stems": [p.stem for p in sample_paths],
        "elapsed_seconds": elapsed,
        "per_image": per_image,
    }

    if args.metrics:
        metrics = evaluate_model_predictions(
            predictions_map,
            ground_truth_map,
            iou_threshold=args.iou,
        )
        results_payload["metrics_vs_pseudo_gt"] = metrics
        print(
            f"mAP@50 vs pseudo-GT: {metrics['mAP']:.4f} "
            f"({metrics['num_images']} images)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"Model: {model_path}")
    print(f"Sampled {sample_size}/{len(all_images)} frames (seed={args.seed})")
    print(f"Annotated images: {annotated_dir}")
    print(f"Results: {results_path}")
    print(f"Total elapsed: {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.data.class_mapping import CANONICAL_CLASSES, load_roboflow_yaml, parse_names_block
from src.utils.evaluation import evaluate_model_predictions

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_yaml(data_arg: str) -> Path:
    path = Path(data_arg)
    if not path.is_absolute():
        path = VISION_ROOT / path
    if path.is_dir():
        for name in ("dataset.yaml", "data.yaml"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No dataset.yaml under {path}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def split_images_dir(yaml_path: Path, split: str) -> Path:
    data = load_roboflow_yaml(yaml_path)
    root = yaml_path.parent
    key = "val" if split == "valid" else split
    rel = data.get(key) or data.get(split)
    if rel is None:
        raise KeyError(f"Split '{split}' not found in {yaml_path}")
    images_dir = (root / rel).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory missing: {images_dir}")
    return images_dir


def yolo_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if parts[-1] == "images":
        return Path(*parts[:-1]) / "labels"
    return images_dir.parent / "labels"


def load_yolo_boxes(
    label_path: Path,
    img_w: int,
    img_h: int,
    id_to_name: Dict[int, str],
) -> List[dict]:
    boxes: List[dict] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            name = id_to_name.get(cid, f"class_{cid}")
            boxes.append({
                "class": name,
                "bbox": [x1, y1, x2, y2],
                "confidence": 1.0,
            })
    return boxes


def load_ground_truth_split(
    yaml_path: Path,
    split: str = "test",
) -> Dict[str, List[dict]]:
    data = load_roboflow_yaml(yaml_path)
    id_to_name = parse_names_block(data)
    images_dir = split_images_dir(yaml_path, split)
    labels_dir = yolo_labels_dir(images_dir)
    ground_truth: Dict[str, List[dict]] = {}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        ground_truth[img_path.stem] = load_yolo_boxes(label_path, w, h, id_to_name)
    return ground_truth


def predictions_from_supervision(det, id_to_name: Dict[int, str]) -> List[dict]:
    if det is None or len(det) == 0:
        return []
    xyxy = np.asarray(det.xyxy)
    class_ids = np.asarray(det.class_id) if det.class_id is not None else np.zeros(len(det))
    confidences = (
        np.asarray(det.confidence)
        if det.confidence is not None
        else np.ones(len(det))
    )
    out: List[dict] = []
    for i in range(len(det)):
        cid = int(class_ids[i])
        name = id_to_name.get(cid, f"class_{cid}")
        out.append({
            "class": name,
            "bbox": xyxy[i].tolist(),
            "confidence": float(confidences[i]),
        })
    return out


def evaluate_yolo(
    model_path: str,
    yaml_path: Path,
    split: str,
    conf: float,
    iou: float,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(yaml_path),
        split=split,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    results = metrics.results_dict if hasattr(metrics, "results_dict") else {}
    map50 = float(results.get("metrics/mAP50(B)", results.get("metrics/mAP50", 0.0)))
    map5095 = float(
        results.get("metrics/mAP50-95(B)", results.get("metrics/mAP50-95", 0.0))
    )
    precision = float(results.get("metrics/precision(B)", 0.0))
    recall = float(results.get("metrics/recall(B)", 0.0))
    return {
        "model": "YOLO",
        "weights": str(model_path),
        "split": split,
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": precision,
        "recall": recall,
        "eval_seconds": round(elapsed, 2),
        "status": "ok",
    }


def evaluate_rfdetr_on_split(
    model_path: Optional[str],
    model_size: str,
    yaml_path: Path,
    split: str,
    conf: float,
    iou: float,
) -> dict:
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    size_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    if model_path:
        raise NotImplementedError(
            "RF-DETR evaluation from custom checkpoint path is not wired yet; "
            "use --rfdetr-size with a trained checkpoint via rfdetr.train output_dir."
        )
    model_cls = size_map.get(model_size.lower(), RFDETRBase)
    model = model_cls()
    id_to_name = {i: n for i, n in enumerate(CANONICAL_CLASSES)}
    images_dir = split_images_dir(yaml_path, split)
    labels_dir = yolo_labels_dir(images_dir)
    predictions: Dict[str, List[dict]] = {}
    ground_truth: Dict[str, List[dict]] = {}
    t0 = time.perf_counter()
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        ground_truth[img_path.stem] = load_yolo_boxes(
            labels_dir / f"{img_path.stem}.txt", w, h, id_to_name
        )
        det = model.predict(str(img_path), threshold=conf)
        predictions[img_path.stem] = predictions_from_supervision(det, id_to_name)
    metrics = evaluate_model_predictions(predictions, ground_truth, iou_threshold=iou)
    elapsed = time.perf_counter() - t0
    return {
        "model": "RF-DETR",
        "size": model_size,
        "split": split,
        "mAP50": metrics["mAP"],
        "mAP50_95": None,
        "precision": None,
        "recall": None,
        "num_images": metrics["num_images"],
        "eval_seconds": round(elapsed, 2),
        "status": "ok",
        "note": "mAP50 via utils.evaluation (11-point); not identical to COCO mAP50-95",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO and RF-DETR on the same frozen dataset split"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chess-pieces-2/dataset.yaml",
        help="Path to dataset.yaml or processed dataset directory",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO .pt weights")
    parser.add_argument("--rfdetr-size", type=str, default=None, choices=["nano", "small", "base", "medium", "large"])
    parser.add_argument("--rfdetr-model", type=str, default=None, help="Reserved for future checkpoint loading")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="JSON output path (relative to vision/)",
    )
    args = parser.parse_args()

    yaml_path = resolve_dataset_yaml(args.data)
    split = "val" if args.split == "valid" else args.split
    results: Dict[str, dict] = {
        "dataset_yaml": str(yaml_path.resolve()),
        "split": split,
        "conf": args.conf,
        "iou": args.iou,
    }

    if args.yolo_model:
        yolo_path = Path(args.yolo_model)
        if not yolo_path.is_absolute():
            yolo_path = VISION_ROOT / yolo_path
        print(f"Evaluating YOLO: {yolo_path}")
        results["yolo"] = evaluate_yolo(str(yolo_path), yaml_path, split, args.conf, args.iou)
        print(f"  mAP50={results['yolo']['mAP50']:.4f}  mAP50-95={results['yolo']['mAP50_95']:.4f}")

    if args.rfdetr_size or args.rfdetr_model:
        size = args.rfdetr_size or "base"
        print(f"Evaluating RF-DETR ({size}) on split={split}")
        results["rfdetr"] = evaluate_rfdetr_on_split(
            args.rfdetr_model, size, yaml_path, split, args.conf, args.iou
        )
        print(f"  mAP50={results['rfdetr']['mAP50']:.4f}  images={results['rfdetr']['num_images']}")

    if "yolo" not in results and "rfdetr" not in results:
        parser.error("Provide at least one of --yolo-model or --rfdetr-size")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = VISION_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

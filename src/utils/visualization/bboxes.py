from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.data.class_mapping import CANONICAL_CLASSES

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "white-pawn": (80, 200, 255),
    "white-rook": (60, 180, 255),
    "white-knight": (40, 160, 255),
    "white-bishop": (20, 140, 255),
    "white-queen": (0, 120, 255),
    "white-king": (0, 100, 255),
    "black-pawn": (180, 180, 180),
    "black-rook": (140, 140, 140),
    "black-knight": (100, 100, 100),
    "black-bishop": (70, 70, 70),
    "black-queen": (40, 40, 40),
    "black-king": (20, 20, 20),
}


def load_yolo_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
    id_to_name: Optional[Dict[int, str]] = None,
) -> List[dict]:
    if id_to_name is None:
        id_to_name = {i: n for i, n in enumerate(CANONICAL_CLASSES)}
    boxes: List[dict] = []
    if not label_path.is_file():
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


def draw_bbox_detections(
    image: np.ndarray,
    detections: Sequence[dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    out = image.copy()
    colors = class_colors or DEFAULT_CLASS_COLORS
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        class_name = str(det.get("class", "unknown"))
        conf = det.get("confidence")
        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if conf is not None and float(conf) < 1.0:
            label = f"{class_name} {float(conf):.2f}"
        else:
            label = class_name
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ly = max(y1, lh + 6)
        cv2.rectangle(out, (x1, ly - lh - 6), (x1 + lw, ly), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(
            out,
            label,
            (x1, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return out


def annotate_image_with_yolo_labels(
    image_path: Path | str,
    label_path: Path | str,
    id_to_name: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    image_path = Path(image_path)
    label_path = Path(label_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    detections = load_yolo_label_file(label_path, w, h, id_to_name=id_to_name)
    return draw_bbox_detections(image, detections, class_colors=class_colors)


def plot_random_dataset_samples(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    n: int = 4,
    seed: int = 42,
    id_to_name: Optional[Dict[int, str]] = None,
) -> List[Path]:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rng = random.Random(seed)
    sample_count = min(n, len(image_paths))
    chosen = rng.sample(image_paths, sample_count)

    saved: List[Path] = []
    for img_path in chosen:
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotated = annotate_image_with_yolo_labels(
            img_path,
            label_path,
            id_to_name=id_to_name,
        )
        out_path = output_dir / f"{img_path.stem}_annotated.jpg"
        if not cv2.imwrite(str(out_path), annotated):
            raise RuntimeError(f"Failed to write {out_path}")
        saved.append(out_path)
    return saved

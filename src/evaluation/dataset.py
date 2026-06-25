from __future__ import annotations

from pathlib import Path

import cv2

from src.data.class_mapping import load_roboflow_yaml, parse_names_block
from src.evaluation.types import Detection, SplitData

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_yaml(data_arg: str | Path, root: Path | None = None) -> Path:
    path = Path(data_arg)
    if root is not None and not path.is_absolute():
        path = root / path
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


def load_yolo_detections(
    label_path: Path,
    img_w: int,
    img_h: int,
    id_to_name: dict[int, str],
) -> list[Detection]:
    detections: list[Detection] = []
    if not label_path.exists():
        return detections
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
            detections.append(
                Detection(
                    class_name=name,
                    bbox=[x1, y1, x2, y2],
                    confidence=1.0,
                )
            )
    return detections


def load_split_data(yaml_path: Path, split: str) -> SplitData:
    data = load_roboflow_yaml(yaml_path)
    id_to_name = parse_names_block(data)
    name_to_id = {name: cid for cid, name in id_to_name.items()}
    images_dir = split_images_dir(yaml_path, split)
    labels_dir = yolo_labels_dir(images_dir)

    image_paths: dict[str, Path] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    ground_truth: dict[str, list[Detection]] = {}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = img_path.stem
        image_paths[stem] = img_path
        image_sizes[stem] = (w, h)
        label_path = labels_dir / f"{stem}.txt"
        ground_truth[stem] = load_yolo_detections(label_path, w, h, id_to_name)

    return SplitData(
        image_paths=image_paths,
        image_sizes=image_sizes,
        ground_truth=ground_truth,
        id_to_name=id_to_name,
        name_to_id=name_to_id,
    )


def detections_to_legacy_dict(detections: list[Detection]) -> list[dict]:
    return [d.to_dict() for d in detections]


def legacy_dict_to_detections(items: list[dict]) -> list[Detection]:
    return [
        Detection(
            class_name=str(item["class"]),
            bbox=[float(v) for v in item["bbox"]],
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in items
    ]

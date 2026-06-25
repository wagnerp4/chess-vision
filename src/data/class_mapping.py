from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import shutil
import yaml

CANONICAL_CLASSES: List[str] = [
    "white-pawn",
    "white-rook",
    "white-knight",
    "white-bishop",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-rook",
    "black-knight",
    "black-bishop",
    "black-queen",
    "black-king",
]

CANONICAL_CLASS_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(CANONICAL_CLASSES)
}

KAGGLE_ALIASES: Dict[str, str] = {
    "white-camel": "white-bishop",
    "white-elephant": "white-rook",
    "white-horse": "white-knight",
    "black-camel": "black-bishop",
    "black-elephant": "black-rook",
    "black-horse": "black-knight",
}

SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}


def normalize_class_name(name: str, legacy_kaggle_map: bool = True) -> Optional[str]:
    key = name.strip().lower().replace("_", "-")
    key = re.sub(r"\s+", "-", key)
    if legacy_kaggle_map and key in KAGGLE_ALIASES:
        key = KAGGLE_ALIASES[key]
    if key in CANONICAL_CLASS_TO_ID:
        return key
    return None


def build_id_remap(
    source_names: Dict[int, str],
    legacy_kaggle_map: bool = True,
) -> Tuple[Dict[int, int], List[str]]:
    remap: Dict[int, int] = {}
    unmapped: List[str] = []
    for old_id, raw_name in source_names.items():
        canonical = normalize_class_name(raw_name, legacy_kaggle_map=legacy_kaggle_map)
        if canonical is None:
            unmapped.append(raw_name)
            continue
        remap[int(old_id)] = CANONICAL_CLASS_TO_ID[canonical]
    return remap, unmapped


def load_roboflow_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {yaml_path}")
    return data


def parse_names_block(data: dict) -> Dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError("data.yaml missing or invalid 'names' field")


def discover_splits(root: Path) -> List[Tuple[str, Path, Path]]:
    splits: List[Tuple[str, Path, Path]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        key = SPLIT_ALIASES.get(folder.name.lower())
        if key is None:
            continue
        images = folder / "images"
        labels = folder / "labels"
        if images.is_dir() and labels.is_dir():
            splits.append((key, images, labels))
    return splits


def remap_label_file(
    src: Path,
    dst: Path,
    id_remap: Dict[int, int],
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    lines_out: List[str] = []
    if not src.exists():
        return 0
    with open(src, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id not in id_remap:
                continue
            parts[0] = str(id_remap[old_id])
            lines_out.append(" ".join(parts))
            kept += 1
    with open(dst, "w") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")
    return kept


def write_dataset_yaml(
    output_root: Path,
    split_keys: Iterable[str],
) -> Path:
    yaml_path = output_root / "dataset.yaml"
    lines = [f"path: {output_root.resolve()}", ""]
    split_paths = {
        "train": "train/images",
        "valid": "valid/images",
        "test": "test/images",
    }
    for key in ("train", "valid", "test"):
        if key in split_keys and (output_root / key / "images").is_dir():
            yaml_key = "val" if key == "valid" else key
            lines.append(f"{yaml_key}: {split_paths[key]}")
    lines.append("")
    lines.append("names:")
    for idx, name in enumerate(CANONICAL_CLASSES):
        lines.append(f"  {idx}: {name}")
    lines.append("")
    content = "\n".join(lines)
    with open(yaml_path, "w") as f:
        f.write(content)
    data_yaml_path = output_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def normalize_dataset_tree(
    input_dir: Path,
    output_dir: Path,
    legacy_kaggle_map: bool = True,
    symlink_images: bool = False,
) -> Path:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    yaml_candidates = list(input_dir.glob("data.yaml")) + list(input_dir.glob("dataset.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError(f"No data.yaml or dataset.yaml under {input_dir}")
    source_yaml = yaml_candidates[0]
    data = load_roboflow_yaml(source_yaml)
    source_names = parse_names_block(data)
    id_remap, unmapped = build_id_remap(source_names, legacy_kaggle_map=legacy_kaggle_map)
    if unmapped:
        raise ValueError(
            f"Unmapped class names (update class_mapping.py or disable strict mode): {unmapped}"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = discover_splits(input_dir)
    if not splits:
        raise FileNotFoundError(f"No train/valid/test splits found under {input_dir}")
    split_keys: List[str] = []
    for split_name, images_dir, labels_dir in splits:
        out_images = output_dir / split_name / "images"
        out_labels = output_dir / split_name / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        split_keys.append(split_name)
        for label_path in labels_dir.glob("*.txt"):
            stem = label_path.stem
            src_img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                continue
            dst_img = out_images / src_img.name
            if symlink_images:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)
            remap_label_file(
                label_path,
                out_labels / label_path.name,
                id_remap,
            )
    return write_dataset_yaml(output_dir, split_keys)

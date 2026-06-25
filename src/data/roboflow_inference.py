from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.data.class_mapping import CANONICAL_CLASS_TO_ID, normalize_class_name

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_MODEL_ID = "chess-pieces-new/19"


def roboflow_detection_to_yolo_line(
    pred: dict,
    img_w: int,
    img_h: int,
    min_confidence: float = 0.0,
) -> Optional[str]:
    confidence = float(pred.get("confidence", 0.0))
    if confidence < min_confidence:
        return None
    raw_class = str(pred.get("class", ""))
    canonical = normalize_class_name(raw_class)
    if canonical is None:
        return None
    if img_w <= 0 or img_h <= 0:
        return None
    xc = float(pred["x"]) / img_w
    yc = float(pred["y"]) / img_h
    w = float(pred["width"]) / img_w
    h = float(pred["height"]) / img_h
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    class_id = CANONICAL_CLASS_TO_ID[canonical]
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def extract_predictions(result: dict) -> List[dict]:
    if not isinstance(result, dict):
        return []
    preds = result.get("predictions", [])
    if isinstance(preds, list):
        return preds
    return []


def extract_image_size(result: dict, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    image_info = result.get("image", {})
    if isinstance(image_info, dict):
        w = int(image_info.get("width", fallback_w))
        h = int(image_info.get("height", fallback_h))
        if w > 0 and h > 0:
            return w, h
    return fallback_w, fallback_h


def predictions_to_label_file(
    predictions: List[dict],
    img_w: int,
    img_h: int,
    out_path: Path,
    min_confidence: float = 0.0,
    unmapped_classes: Optional[Set[str]] = None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for pred in predictions:
        raw_class = str(pred.get("class", ""))
        canonical = normalize_class_name(raw_class)
        if canonical is None:
            if unmapped_classes is not None:
                unmapped_classes.add(raw_class)
            continue
        line = roboflow_detection_to_yolo_line(pred, img_w, img_h, min_confidence=min_confidence)
        if line is not None:
            lines.append(line)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


def build_inference_client(api_key: str, api_url: str = DEFAULT_API_URL):
    from inference_sdk import InferenceHTTPClient

    return InferenceHTTPClient(api_url=api_url, api_key=api_key)


def infer_image(
    client,
    image_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    confidence: float = 0.25,
) -> dict:
    result = client.infer(str(image_path), model_id=model_id)
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected inference response type: {type(result)}")
    preds = extract_predictions(result)
    if confidence > 0.0:
        preds = [p for p in preds if float(p.get("confidence", 0.0)) >= confidence]
        result = dict(result)
        result["predictions"] = preds
    return result


def load_cached_prediction(cache_path: Path) -> Optional[dict]:
    if not cache_path.is_file():
        return None
    with open(cache_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return None


def save_cached_prediction(cache_path: Path, result: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)


def infer_image_cached(
    client,
    image_path: Path,
    cache_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    confidence: float = 0.25,
) -> dict:
    cached = load_cached_prediction(cache_path)
    if cached is not None:
        return cached
    result = infer_image(client, image_path, model_id=model_id, confidence=confidence)
    save_cached_prediction(cache_path, result)
    return result

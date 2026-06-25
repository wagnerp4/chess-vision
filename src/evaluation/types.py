from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Detection:
    class_name: str
    bbox: list[float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EvalConfig:
    conf: float = 0.25
    iou_match: float = 0.5
    iou_nms: float = 0.45
    imgsz: int = 640
    max_det: int = 300


@dataclass(frozen=True)
class BenchmarkScenario:
    id: str
    dataset_yaml: str
    split: str
    label_type: Literal["human", "pseudo"] = "human"
    label: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    weights: str
    label: str
    size: str | None = None


@dataclass
class BenchmarkManifest:
    eval: EvalConfig
    scenarios: list[BenchmarkScenario]
    models: list[ModelSpec]


@dataclass
class SplitData:
    image_paths: dict[str, Path]
    image_sizes: dict[str, tuple[int, int]]
    ground_truth: dict[str, list[Detection]]
    id_to_name: dict[int, str]
    name_to_id: dict[str, int]


@dataclass
class EvalResult:
    model: dict[str, Any]
    scenario: dict[str, Any]
    eval_config: dict[str, Any]
    metrics: dict[str, Any]
    num_images: int
    eval_seconds: float
    ms_per_image: float
    status: str = "ok"
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "eval_config": self.eval_config,
            "metrics": self.metrics,
            "num_images": self.num_images,
            "eval_seconds": self.eval_seconds,
            "ms_per_image": self.ms_per_image,
            "status": self.status,
            "per_class": self.per_class,
        }

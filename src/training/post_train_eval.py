from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation import (
    BenchmarkScenario,
    EvalConfig,
    ModelSpec,
    resolve_dataset_yaml,
    run_scenario,
)
from src.training.paths import anchor_path
from src.training.run_mode import RunMode, filter_post_train_entries


def run_post_train_eval(
    config: dict,
    checkpoint_path: Path,
    model_size: str,
    root: Path | None = None,
    run_mode: RunMode = "full",
) -> dict[str, Any]:
    evaluation = config.get("evaluation") or {}
    conf = float(evaluation.get("conf_threshold", 0.25))
    iou_match = float(evaluation.get("iou_threshold", 0.5))
    iou_nms = float(evaluation.get("iou_nms", 0.45))
    imgsz = int(evaluation.get("imgsz", 640))
    max_det = int(evaluation.get("max_det", 300))
    eval_cfg = EvalConfig(
        conf=conf,
        iou_match=iou_match,
        iou_nms=iou_nms,
        imgsz=imgsz,
        max_det=max_det,
    )
    entries = filter_post_train_entries(config, run_mode, root)

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "model_size": model_size,
        "run_mode": run_mode,
        "eval_config": eval_cfg.__dict__,
        "splits": {},
    }

    for entry in entries:
        yaml_path = resolve_dataset_yaml(str(anchor_path(entry["dataset_path"], root)), root=root)
        split = entry.get("split", "val")
        label = entry.get("label") or f"{yaml_path.parent.name}_{split}"
        print(f"Post-train eval: {label} split={split} checkpoint={checkpoint_path.name}")
        result = run_scenario(
            ModelSpec(
                id=f"rfdetr_{checkpoint_path.stem}",
                backend="rfdetr",
                weights=str(checkpoint_path),
                label=f"RF-DETR {model_size}",
                size=model_size,
            ),
            BenchmarkScenario(
                id=label,
                dataset_yaml=str(yaml_path),
                split=split,
                label_type=entry.get("label_type", "human"),
                label=label,
            ),
            eval_cfg,
            root=root,
        )
        manifest["splits"][label] = result.to_dict()
        print(
            f"  COCO mAP50={result.metrics['coco']['mAP50']:.4f}  "
            f"legacy mAP50={result.metrics['legacy_ap']['mAP50']:.4f}  "
            f"images={result.num_images}"
        )

    return manifest


def write_benchmark_json(
    manifest: dict[str, Any],
    config: dict,
    root: Path | None = None,
    run_mode: RunMode = "full",
) -> Path:
    best_dir = anchor_path("data/checkpoints/best", root)
    best_dir.mkdir(parents=True, exist_ok=True)
    run_name = config["training"]["name"]
    suffix = ""
    if run_mode == "fast_dev":
        suffix = "_fastdev"
    elif run_mode == "smoke":
        suffix = "_smoke"
    out_path = best_dir / f"benchmark_{run_name}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Post-train benchmark saved to {out_path}")
    return out_path

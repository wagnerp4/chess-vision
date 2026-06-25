from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.backends.registry import get_backend
from src.evaluation.coco_eval import evaluate_dual_metrics
from src.evaluation.dataset import load_split_data, resolve_dataset_yaml
from src.evaluation.types import (
    BenchmarkManifest,
    BenchmarkScenario,
    EvalConfig,
    EvalResult,
    ModelSpec,
)


def _normalize_split(split: str) -> str:
    return "val" if split == "valid" else split


def load_benchmark_manifest(manifest_path: Path, root: Path | None = None) -> BenchmarkManifest:
    path = Path(manifest_path)
    if root is not None and not path.is_absolute():
        path = root / path
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    eval_raw = raw.get("eval", {})
    eval_cfg = EvalConfig(
        conf=float(eval_raw.get("conf", 0.25)),
        iou_match=float(eval_raw.get("iou_match", 0.5)),
        iou_nms=float(eval_raw.get("iou_nms", 0.45)),
        imgsz=int(eval_raw.get("imgsz", 640)),
        max_det=int(eval_raw.get("max_det", 300)),
    )

    scenarios: list[BenchmarkScenario] = []
    for entry in raw.get("scenarios", []):
        scenarios.append(
            BenchmarkScenario(
                id=str(entry["id"]),
                dataset_yaml=str(entry["dataset_yaml"]),
                split=_normalize_split(str(entry.get("split", "test"))),
                label_type=entry.get("label_type", "human"),
                label=entry.get("label"),
            )
        )

    models: list[ModelSpec] = []
    for entry in raw.get("models", []):
        models.append(
            ModelSpec(
                id=str(entry["id"]),
                backend=str(entry["backend"]),
                weights=str(entry["weights"]),
                label=str(entry.get("label", entry["id"])),
                size=entry.get("size"),
            )
        )

    return BenchmarkManifest(eval=eval_cfg, scenarios=scenarios, models=models)


def manifest_hash(manifest: BenchmarkManifest) -> str:
    payload = json.dumps(
        {
            "eval": manifest.eval.__dict__,
            "scenarios": [s.__dict__ for s in manifest.scenarios],
            "models": [m.__dict__ for m in manifest.models],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_scenario(
    model_spec: ModelSpec,
    scenario: BenchmarkScenario,
    eval_cfg: EvalConfig,
    root: Path | None = None,
) -> EvalResult:
    yaml_path = resolve_dataset_yaml(scenario.dataset_yaml, root=root)
    split_data = load_split_data(yaml_path, scenario.split)

    backend = get_backend(model_spec)
    backend.load(model_spec, root=root)
    if hasattr(backend, "set_class_names"):
        backend.set_class_names(split_data.id_to_name)

    predictions: dict[str, list] = {}
    t0 = time.perf_counter()
    for stem, img_path in sorted(split_data.image_paths.items()):
        predictions[stem] = backend.predict(img_path, eval_cfg)
    elapsed = time.perf_counter() - t0

    coco_metrics, legacy_metrics, per_class = evaluate_dual_metrics(
        split_data,
        predictions,
        eval_cfg.iou_match,
    )

    num_images = len(split_data.image_paths)
    ms_per_image = (elapsed * 1000.0 / num_images) if num_images else 0.0

    weights_path = Path(model_spec.weights)
    if root is not None and not weights_path.is_absolute():
        weights_path = root / weights_path

    return EvalResult(
        model={
            "id": model_spec.id,
            "backend": model_spec.backend,
            "weights": str(weights_path.resolve()),
            "label": model_spec.label,
            "size": model_spec.size,
        },
        scenario={
            "id": scenario.id,
            "dataset_yaml": str(yaml_path.resolve()),
            "split": scenario.split,
            "label_type": scenario.label_type,
            "label": scenario.label or scenario.id,
        },
        eval_config={
            "conf": eval_cfg.conf,
            "iou_match": eval_cfg.iou_match,
            "iou_nms": eval_cfg.iou_nms,
            "imgsz": eval_cfg.imgsz,
            "max_det": eval_cfg.max_det,
            "leaderboard_primary": "metrics.coco.mAP50",
        },
        metrics={
            "coco": {
                "mAP50": coco_metrics["mAP50"],
                "mAP50_95": coco_metrics["mAP50_95"],
                "precision": coco_metrics["precision"],
                "recall": coco_metrics["recall"],
            },
            "legacy_ap": legacy_metrics,
        },
        num_images=num_images,
        eval_seconds=round(elapsed, 2),
        ms_per_image=round(ms_per_image, 2),
        per_class=per_class,
    )


def run_benchmark(
    manifest_path: Path,
    root: Path | None = None,
    model_ids: list[str] | None = None,
    scenario_ids: list[str] | None = None,
) -> list[EvalResult]:
    manifest = load_benchmark_manifest(manifest_path, root=root)
    models = manifest.models
    scenarios = manifest.scenarios

    if model_ids:
        models = [m for m in models if m.id in model_ids]
    if scenario_ids:
        scenarios = [s for s in scenarios if s.id in scenario_ids]

    results: list[EvalResult] = []
    for model_spec in models:
        for scenario in scenarios:
            print(f"Evaluating {model_spec.label} on {scenario.id} …")
            result = run_scenario(model_spec, scenario, manifest.eval, root=root)
            results.append(result)
            print(
                f"  COCO mAP50={result.metrics['coco']['mAP50']:.4f}  "
                f"legacy mAP50={result.metrics['legacy_ap']['mAP50']:.4f}  "
                f"images={result.num_images}"
            )
    return results


def save_results(
    results: list[EvalResult],
    out_dir: Path,
    manifest_path: Path,
    manifest: BenchmarkManifest,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mhash = manifest_hash(manifest)
    summary_path = out_dir / "leaderboard.json"
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_hash": mhash,
        "eval_config": manifest.eval.__dict__,
        "results": [r.to_dict() for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    for result in results:
        model_id = result.model["id"]
        scenario_id = result.scenario["id"]
        cell_dir = out_dir / model_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_path = cell_dir / f"{scenario_id}.json"
        with open(cell_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return summary_path


def write_leaderboard_md(results: list[EvalResult], out_path: Path) -> None:
    scenarios = sorted({r.scenario["id"] for r in results}, key=str)

    def pct(x: float | None) -> str:
        if x is None:
            return "—"
        return f"{100.0 * x:.1f}%"

    eval_cfg = results[0].eval_config if results else {}

    lines = [
        "# Detector benchmark leaderboard",
        "",
        "Unified **predict-then-score** evaluation. Every backend runs inference on the same "
        "frozen split, then metrics are computed by shared scorers (no backend-specific `val()`).",
        "",
        "## Protocol",
        "",
        "- Manifest: `configs/benchmark.yaml` (datasets, splits, models, thresholds)",
        "- Primary metric: **COCO mAP@50** (`metrics.coco.mAP50`)",
        "- Secondary: COCO mAP@50:95, legacy 11-point per-image AP (`metrics.legacy_ap.mAP50`)",
        f"- Eval config: `conf={eval_cfg.get('conf')}`, `iou_match={eval_cfg.get('iou_match')}`, "
        f"`iou_nms={eval_cfg.get('iou_nms')}`, `imgsz={eval_cfg.get('imgsz')}`",
        "",
        "Regenerate:",
        "",
        "```bash",
        "cd Vision/Personal/chess-vision",
        "uv run python scripts/evaluate_models.py --manifest configs/benchmark.yaml",
        "```",
        "",
        "Results JSON: `results/benchmark/leaderboard.json` plus per-cell files under "
        "`results/benchmark/<model_id>/`.",
        "",
        "## Notes",
        "",
        "- `chess_pieces2_test` is the official Roboflow held-out split (16 images, high variance).",
        "- `chess_pieces2_val` is the stable in-domain split (910 images).",
        "- `neuroTUM_test` uses pseudo-GT from Roboflow teacher labels (810 frames, domain shift).",
        "- Add a new detector: implement `DetectorBackend` under `src/evaluation/backends/` and "
        "register in `registry.py`.",
        "",
    ]

    for scenario_id in scenarios:
        scenario_rows = [r for r in results if r.scenario["id"] == scenario_id]
        if not scenario_rows:
            continue
        label_type = scenario_rows[0].scenario.get("label_type", "human")
        scenario_label = scenario_rows[0].scenario.get("label", scenario_id)
        lines.append(f"## {scenario_id} (label_type={label_type})")
        lines.append("")
        lines.append(f"_{scenario_label}_")
        lines.append("")
        lines.append("| Model | COCO mAP@50 | COCO mAP@50:95 | Legacy mAP@50 | Images |")
        lines.append("|-------|-------------|----------------|---------------|--------|")
        for result in sorted(
            scenario_rows,
            key=lambda r: r.metrics["coco"]["mAP50"],
            reverse=True,
        ):
            coco = result.metrics["coco"]
            legacy = result.metrics["legacy_ap"]
            lines.append(
                f"| {result.model['label']} | "
                f"{pct(coco['mAP50'])} | {pct(coco['mAP50_95'])} | "
                f"{pct(legacy['mAP50'])} | {result.num_images} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

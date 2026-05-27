import argparse
import json
import sys
from pathlib import Path
from typing import Any

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

import time

from scripts.evaluate_models import resolve_dataset_yaml

OFFICIAL_ROBOFLOW = {
    "label": "Roboflow 3.0 (chess-pieces-2 v5, Universe)",
    "mAP50": 0.962,
    "precision": 0.968,
    "recall": 0.944,
}

CHECKPOINTS: list[dict[str, Any]] = [
    {
        "label": "best_20260110_122310 (OAK export, not 12-class)",
        "weights": "data/checkpoints/best/best_20260110_122310.pt",
        "run_dir": None,
    },
    {
        "label": "best_20260515_yolo_chess_pieces_2 (YOLOv8n)",
        "weights": "data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt",
        "run_dir": None,
    },
    {
        "label": "best_20260527_yolo26n_chess_pieces_2 (YOLO26n)",
        "weights": "data/checkpoints/best/best_20260527_yolo26n_chess_pieces_2.pt",
        "run_dir": "data/runs/detect/yolo26n_chess_pieces_2-2",
    },
]


def evaluate_yolo_with_per_class(
    model_path: str,
    yaml_path: Path,
    split: str,
    conf: float | None,
    iou: float,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(yaml_path),
        split=split,
        conf=conf,
        iou=iou,
        verbose=False,
        plots=False,
    )
    elapsed = time.perf_counter() - t0
    results = metrics.results_dict if hasattr(metrics, "results_dict") else {}
    summary = {
        "model": "YOLO",
        "weights": str(model_path),
        "split": split,
        "mAP50": float(results.get("metrics/mAP50(B)", results.get("metrics/mAP50", 0.0))),
        "mAP50_95": float(
            results.get("metrics/mAP50-95(B)", results.get("metrics/mAP50-95", 0.0))
        ),
        "precision": float(results.get("metrics/precision(B)", 0.0)),
        "recall": float(results.get("metrics/recall(B)", 0.0)),
        "eval_seconds": round(elapsed, 2),
        "status": "ok",
    }

    per_class: dict[str, dict[str, float]] = {}
    names = getattr(metrics, "names", {}) or {}
    box = getattr(metrics, "box", None)
    if box is not None and names:
        ap50 = getattr(box, "ap50", None)
        p_arr = getattr(box, "p", None)
        r_arr = getattr(box, "r", None)
        for idx, name in names.items():
            entry: dict[str, float] = {}
            if ap50 is not None and len(ap50) > idx:
                entry["mAP50"] = float(ap50[idx])
            if p_arr is not None and len(p_arr) > idx:
                entry["precision"] = float(p_arr[idx])
            if r_arr is not None and len(r_arr) > idx:
                entry["recall"] = float(r_arr[idx])
            per_class[str(name)] = entry

    summary["per_class"] = per_class
    summary["metrics_raw"] = {k: float(v) for k, v in results.items() if isinstance(v, (int, float))}
    return summary


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def write_benchmark_md(
    rows: list[dict[str, Any]],
    split: str,
    dataset_yaml: str,
    conf: float | None,
    iou_nms: float,
) -> str:
    lines = [
        "# Chess Pieces 2 — detection benchmark",
        "",
        "Comparison on the **Roboflow Chess Pieces 2** processed dataset "
        f"(`{dataset_yaml}`), split **`{split}`** (910 images). "
        f"Ultralytics `model.val()` with `conf={conf!r}`, `iou={iou_nms}` (NMS). "
        "End-of-train logs use the same split and `conf=None` unless you changed training args.",
        "",
        "**Official Roboflow 3.0** scores (Universe model card, v5) are the reference row.",
        "",
        "| Model | mAP@50 | Precision | Recall |",
        "|-------|--------|-----------|--------|",
        f"| {OFFICIAL_ROBOFLOW['label']} | "
        f"{pct(OFFICIAL_ROBOFLOW['mAP50'])} | "
        f"{pct(OFFICIAL_ROBOFLOW['precision'])} | "
        f"{pct(OFFICIAL_ROBOFLOW['recall'])} |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {pct(row.get('mAP50'))} | "
            f"{pct(row.get('precision'))} | {pct(row.get('recall'))} |"
        )
    lines.extend([
        "",
        "## Per-class metrics (local checkpoints)",
        "",
        "Class-wise scores are written next to each checkpoint run:",
        "",
        "- `data/runs/detect/yolo26n_chess_pieces_2-2/benchmark_val.json` (YOLO26n run)",
        "- `data/checkpoints/best/benchmark_<checkpoint_stem>.json` for checkpoints without a linked run dir",
        "",
        "Regenerate this table:",
        "",
        "```bash",
        "cd vision",
        "uv run python scripts/benchmark_yolo_checkpoints.py --split val",
        "```",
        "",
        "## Notes",
        "",
        "- `infer_yolo.py` is for **single-image** inference and visualization only.",
        "- Use `scripts/evaluate_models.py` or this script for **mAP@50 / P / R**.",
        "- **Split does not change** between train-end val and this table: both use Roboflow `valid` → YAML key `val` (910 images, 57 batches at batch 16).",
        "- **92.3% vs 94.8%:** the first benchmark run used `conf=0.25` (from `configs/yolo.yaml` inference defaults). Training val uses `conf=None` (low threshold for mAP). Re-run with `--conf none` to match the training log.",
        "- `iou=0.7` in training is **NMS** IoU, not the 0.5 used inside mAP@50 box matching.",
        "- Roboflow official numbers may use their own conf/NMS. Treat gaps as indicative.",
        "- `best_20260110_122310` scores ~0% on this 12-class layout (likely OAK export, not comparable).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark archived YOLO checkpoints vs Roboflow official scores"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chess-pieces-2/dataset.yaml",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "valid", "test"])
    parser.add_argument(
        "--conf",
        type=str,
        default="none",
        help="Confidence threshold for val (none = Ultralytics default, matches end-of-train val)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU during val (training used 0.7 in args.yaml; mAP@50 still uses 0.5 for matching)",
    )
    parser.add_argument(
        "--write-md",
        type=str,
        default="docs/benchmark.md",
        help="Update benchmark markdown table",
    )
    args = parser.parse_args()
    conf = None if str(args.conf).lower() in ("none", "null", "") else float(args.conf)

    yaml_path = resolve_dataset_yaml(args.data)
    split = "val" if args.split == "valid" else args.split
    table_rows: list[dict[str, Any]] = []

    for entry in CHECKPOINTS:
        weights = VISION_ROOT / entry["weights"]
        if not weights.is_file():
            print(f"Skip missing weights: {weights}")
            continue
        print(f"Evaluating {entry['label']} …")
        result = evaluate_yolo_with_per_class(
            str(weights), yaml_path, split, conf, args.iou
        )
        result["conf"] = conf
        result["iou_nms"] = args.iou
        result["label"] = entry["label"]
        table_rows.append(result)

        out_dir: Path | None = None
        if entry.get("run_dir"):
            run_dir = VISION_ROOT / entry["run_dir"]
            if run_dir.is_dir():
                out_dir = run_dir
        if out_dir is None:
            out_dir = VISION_ROOT / "data/checkpoints/best"
        out_path = out_dir / (
            "benchmark_val.json"
            if entry.get("run_dir") and (VISION_ROOT / entry["run_dir"]).is_dir()
            else f"benchmark_{weights.stem}.json"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"  mAP50={result['mAP50']:.3f}  P={result['precision']:.3f}  "
            f"R={result['recall']:.3f}  → {out_path}"
        )

    md_path = VISION_ROOT / args.write_md
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        write_benchmark_md(
            table_rows,
            split,
            str(yaml_path.relative_to(VISION_ROOT)),
            conf,
            args.iou,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

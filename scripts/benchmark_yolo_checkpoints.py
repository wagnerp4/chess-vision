import argparse
import sys
from pathlib import Path
from typing import Any

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.evaluation import (
    BenchmarkScenario,
    EvalConfig,
    ModelSpec,
    load_benchmark_manifest,
    resolve_dataset_yaml,
    run_scenario,
    write_leaderboard_md,
)

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


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def write_benchmark_md(
    rows: list[dict[str, Any]],
    split: str,
    dataset_yaml: str,
    eval_cfg: EvalConfig,
) -> str:
    lines = [
        "# Chess Pieces 2 — detection benchmark",
        "",
        "Unified predict-then-score evaluation on the **Roboflow Chess Pieces 2** "
        f"processed dataset (`{dataset_yaml}`), split **`{split}`**. "
        f"Eval config: `conf={eval_cfg.conf}`, `iou_match={eval_cfg.iou_match}`, "
        f"`iou_nms={eval_cfg.iou_nms}`. Primary metric: **COCO mAP@50**.",
        "",
        "**Official Roboflow 3.0** scores (Universe model card, v5) are the reference row.",
        "",
        "| Model | COCO mAP@50 | Legacy mAP@50 | COCO P | COCO R |",
        "|-------|-------------|---------------|--------|--------|",
        f"| {OFFICIAL_ROBOFLOW['label']} | "
        f"{pct(OFFICIAL_ROBOFLOW['mAP50'])} | — | "
        f"{pct(OFFICIAL_ROBOFLOW['precision'])} | "
        f"{pct(OFFICIAL_ROBOFLOW['recall'])} |",
    ]
    for row in rows:
        coco = row["metrics"]["coco"]
        legacy = row["metrics"]["legacy_ap"]
        lines.append(
            f"| {row['label']} | {pct(coco.get('mAP50'))} | "
            f"{pct(legacy.get('mAP50'))} | "
            f"{pct(coco.get('precision'))} | {pct(coco.get('recall'))} |"
        )
    lines.extend([
        "",
        "Regenerate via unified benchmark:",
        "",
        "```bash",
        "cd vision",
        "uv run python scripts/evaluate_models.py --manifest configs/benchmark.yaml",
        "```",
        "",
        "Or YOLO-only val split:",
        "",
        "```bash",
        "uv run python scripts/benchmark_yolo_checkpoints.py --split val",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark archived YOLO checkpoints via unified evaluation runner"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chess-pieces-2/dataset.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "valid", "test"],
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5, dest="iou_match")
    parser.add_argument("--iou-nms", type=float, default=0.45, dest="iou_nms")
    parser.add_argument(
        "--write-md",
        type=str,
        default="docs/benchmark.md",
        help="Update benchmark markdown table",
    )
    args = parser.parse_args()

    yaml_path = resolve_dataset_yaml(args.data, root=VISION_ROOT)
    split = "val" if args.split == "valid" else args.split
    eval_cfg = EvalConfig(
        conf=args.conf,
        iou_match=args.iou_match,
        iou_nms=args.iou_nms,
    )
    scenario = BenchmarkScenario(
        id=f"chess_pieces2_{split}",
        dataset_yaml=str(yaml_path.relative_to(VISION_ROOT)),
        split=split,
        label_type="human",
    )

    table_rows: list[dict[str, Any]] = []
    eval_results = []

    for entry in CHECKPOINTS:
        weights = VISION_ROOT / entry["weights"]
        if not weights.is_file():
            print(f"Skip missing weights: {weights}")
            continue
        print(f"Evaluating {entry['label']} …")
        result = run_scenario(
            ModelSpec(
                id=weights.stem,
                backend="ultralytics",
                weights=str(entry["weights"]),
                label=entry["label"],
            ),
            scenario,
            eval_cfg,
            root=VISION_ROOT,
        )
        row = result.to_dict()
        row["label"] = entry["label"]
        table_rows.append(row)
        eval_results.append(result)

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
        import json

        with open(out_path, "w") as f:
            json.dump(row, f, indent=2)
        coco = result.metrics["coco"]
        print(
            f"  COCO mAP50={coco['mAP50']:.3f}  P={coco['precision']:.3f}  "
            f"R={coco['recall']:.3f}  → {out_path}"
        )

    md_path = VISION_ROOT / args.write_md
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        write_benchmark_md(
            table_rows,
            split,
            str(yaml_path.relative_to(VISION_ROOT)),
            eval_cfg,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

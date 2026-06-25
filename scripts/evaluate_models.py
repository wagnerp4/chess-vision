import argparse
import json
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.evaluation import (
    IMG_EXTENSIONS,
    EvalConfig,
    ModelSpec,
    BenchmarkScenario,
    load_split_data,
    load_yolo_detections,
    resolve_dataset_yaml,
    run_benchmark,
    run_scenario,
    save_results,
    split_images_dir,
    write_leaderboard_md,
    yolo_labels_dir,
    load_benchmark_manifest,
)
from src.evaluation.backends.rfdetr import predictions_from_supervision  # re-export for scripts

def load_yolo_boxes(label_path, img_w, img_h, id_to_name):
    dets = load_yolo_detections(label_path, img_w, img_h, id_to_name)
    return [d.to_dict() for d in dets]


def load_ground_truth_split(yaml_path, split="test"):
    split_data = load_split_data(yaml_path, split)
    return {
        stem: [d.to_dict() for d in dets]
        for stem, dets in split_data.ground_truth.items()
    }


def evaluate_rfdetr_on_split(model_path, model_size, yaml_path, split, conf, iou):
    result = run_scenario(
        ModelSpec(
            id="rfdetr_adhoc",
            backend="rfdetr",
            weights=str(model_path) if model_path else "",
            label=f"RF-DETR {model_size}",
            size=model_size,
        ),
        BenchmarkScenario(
            id=f"adhoc_{split}",
            dataset_yaml=str(yaml_path),
            split=split,
        ),
        EvalConfig(conf=conf, iou_match=iou),
        root=VISION_ROOT,
    )
    d = result.to_dict()
    return {
        "model": "RF-DETR",
        "size": model_size,
        "split": split,
        "mAP50": result.metrics["legacy_ap"]["mAP50"],
        "mAP50_95": result.metrics["coco"]["mAP50_95"],
        "precision": result.metrics["coco"]["precision"],
        "recall": result.metrics["coco"]["recall"],
        "num_images": result.num_images,
        "eval_seconds": result.eval_seconds,
        "status": result.status,
        "metrics": result.metrics,
        "note": d["metrics"]["legacy_ap"].get("note"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified backend-agnostic detector benchmark"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to benchmark manifest YAML (runs full matrix)",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Single model id from manifest")
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Single scenario id from manifest",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chess-pieces-2/dataset.yaml",
        help="Dataset yaml for legacy one-off eval",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "valid", "test"],
    )
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO .pt weights")
    parser.add_argument(
        "--rfdetr-size",
        type=str,
        default=None,
        choices=["nano", "small", "base", "medium", "large"],
    )
    parser.add_argument(
        "--rfdetr-model",
        type=str,
        default=None,
        help="Path to fine-tuned RF-DETR checkpoint (.ckpt or .pth)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5, dest="iou_match")
    parser.add_argument("--iou-nms", type=float, default=0.45, dest="iou_nms")
    parser.add_argument("--iou", type=float, default=None, help="Alias for --iou-match")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark",
        help="Output directory or JSON path",
    )
    parser.add_argument(
        "--write-md",
        type=str,
        default="docs/benchmark.md",
        help="Write markdown leaderboard table",
    )
    args = parser.parse_args()

    iou_match = args.iou if args.iou is not None else args.iou_match
    eval_cfg = EvalConfig(
        conf=args.conf,
        iou_match=iou_match,
        iou_nms=args.iou_nms,
        imgsz=args.imgsz,
    )

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = VISION_ROOT / manifest_path
        manifest = load_benchmark_manifest(manifest_path, root=VISION_ROOT)
        model_ids = [args.model_id] if args.model_id else None
        scenario_ids = [args.scenario_id] if args.scenario_id else None
        results = run_benchmark(
            manifest_path,
            root=VISION_ROOT,
            model_ids=model_ids,
            scenario_ids=scenario_ids,
        )
        out_dir = Path(args.output)
        if not out_dir.is_absolute():
            out_dir = VISION_ROOT / out_dir
        summary_path = save_results(results, out_dir, manifest_path, manifest)
        md_path = Path(args.write_md)
        if not md_path.is_absolute():
            md_path = VISION_ROOT / md_path
        write_leaderboard_md(results, md_path)
        print(f"Leaderboard saved to {summary_path}")
        print(f"Markdown table saved to {md_path}")
        return

    results = []
    split = "val" if args.split == "valid" else args.split
    yaml_path = resolve_dataset_yaml(args.data, root=VISION_ROOT)
    scenario = BenchmarkScenario(
        id=f"legacy_{split}",
        dataset_yaml=str(yaml_path.relative_to(VISION_ROOT)),
        split=split,
    )

    if args.yolo_model:
        yolo_path = Path(args.yolo_model)
        if not yolo_path.is_absolute():
            yolo_path = VISION_ROOT / yolo_path
        print(f"Evaluating YOLO: {yolo_path}")
        result = run_scenario(
            ModelSpec(
                id="yolo_adhoc",
                backend="ultralytics",
                weights=str(yolo_path),
                label=str(yolo_path.name),
            ),
            scenario,
            eval_cfg,
            root=VISION_ROOT,
        )
        results.append(result)
        print(
            f"  COCO mAP50={result.metrics['coco']['mAP50']:.4f}  "
            f"legacy mAP50={result.metrics['legacy_ap']['mAP50']:.4f}"
        )

    if args.rfdetr_size or args.rfdetr_model:
        size = args.rfdetr_size or "large"
        weights = args.rfdetr_model or ""
        print(f"Evaluating RF-DETR ({size}) on split={split}")
        result = run_scenario(
            ModelSpec(
                id="rfdetr_adhoc",
                backend="rfdetr",
                weights=weights,
                label=f"RF-DETR {size}",
                size=size,
            ),
            scenario,
            eval_cfg,
            root=VISION_ROOT,
        )
        results.append(result)
        print(
            f"  COCO mAP50={result.metrics['coco']['mAP50']:.4f}  "
            f"legacy mAP50={result.metrics['legacy_ap']['mAP50']:.4f}  "
            f"images={result.num_images}"
        )

    if not results:
        parser.error(
            "Provide --manifest or at least one of --yolo-model / --rfdetr-model"
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = VISION_ROOT / out_path
    if out_path.suffix == ".json":
        payload = {
            "dataset_yaml": str(yaml_path.resolve()),
            "split": split,
            "eval_config": eval_cfg.__dict__,
            "results": [r.to_dict() for r in results],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {out_path}")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        for result in results:
            cell_path = out_path / f"{result.model['id']}_{result.scenario['id']}.json"
            with open(cell_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Results saved to {cell_path}")


if __name__ == "__main__":
    main()

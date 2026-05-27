import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

YOLO26_VARIANTS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Ultralytics pretrained YOLO weights into data/checkpoints/pretrained/"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n",
        help="Model name without .pt (default: yolo26n). Examples: yolo26s, yolo11n, yolov8n",
    )
    parser.add_argument(
        "--list-yolo26",
        action="store_true",
        help="Print YOLO26 size variants and exit",
    )
    args = parser.parse_args()

    if args.list_yolo26:
        for name in YOLO26_VARIANTS:
            print(name)
        return 0

    from recipes.roboflow.finetune.yolo.train import load_yolo_model

    model_name = args.model.strip()
    if model_name.endswith(".pt"):
        model_name = model_name[:-3]

    config = {
        "model": {"name": model_name, "pretrained": True},
        "training": {"name": "pretrained_download"},
    }
    load_yolo_model(config, VISION_ROOT)
    out = VISION_ROOT / "data" / "checkpoints" / "pretrained" / f"{model_name}.pt"
    if not out.exists():
        print(f"Expected weights at {out} but file is missing.", file=sys.stderr)
        return 1
    print(f"Pretrained weights ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

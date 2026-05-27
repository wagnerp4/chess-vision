import argparse
import os
import subprocess
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = VISION_ROOT / "configs" / "roboflow" / "yolo26" / "yolo26.yaml"
DATASET_YAML = VISION_ROOT / "data" / "processed" / "chess-pieces-2" / "dataset.yaml"
ROBOFLOW_RAW = VISION_ROOT / "data" / "roboflow" / "chess-pieces-2"


def _load_env() -> None:
    sys.path.insert(0, str(VISION_ROOT))
    from src.utils.download.roboflow import _load_env_files

    _load_env_files(VISION_ROOT)


def _run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=VISION_ROOT, check=True, env=os.environ.copy())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Chess Pieces 2 (Roboflow), YOLO26 pretrained weights, and fine-tune"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Training YAML (default: configs/roboflow/yolo26/yolo26.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n",
        help="YOLO26 variant for pretrained download (yolo26n/s/m/l/x)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip Roboflow download (still normalizes if raw export exists but processed yaml is missing)",
    )
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip YOLO26 .pt download")
    parser.add_argument("--skip-train", action="store_true", help="Only prepare data and weights")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Re-download Roboflow export")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt to resume training",
    )
    args = parser.parse_args()

    _load_env()

    py = sys.executable

    if not args.skip_dataset:
        dl_cmd = [
            py,
            "scripts/download_roboflow.py",
            "--preset",
            "chess-pieces-2",
        ]
        if args.overwrite_dataset:
            dl_cmd.append("--overwrite")
        _run(dl_cmd, "Download Roboflow Chess Pieces 2 (~6.6k images)")

    if not DATASET_YAML.exists():
        raw_ready = (ROBOFLOW_RAW / "data.yaml").exists() and (ROBOFLOW_RAW / "train").exists()
        if args.skip_dataset and not raw_ready:
            print(
                f"Missing {DATASET_YAML} and no raw export at {ROBOFLOW_RAW}. "
                "Run without --skip-dataset first.",
                file=sys.stderr,
            )
            return 1
        if raw_ready:
            _run(
                [
                    py,
                    "scripts/normalize_dataset.py",
                    "--input",
                    str(ROBOFLOW_RAW),
                ],
                "Normalize to data/processed/chess-pieces-2/",
            )

    if not DATASET_YAML.exists():
        print(f"Missing {DATASET_YAML} after normalize.", file=sys.stderr)
        return 1

    if not args.skip_pretrained:
        _run(
            [py, "scripts/download_yolo_pretrained.py", "--model", args.model],
            f"Download pretrained {args.model}",
        )

    if args.skip_train:
        print("\nSkipping training (--skip-train).")
        print(f"Dataset: {DATASET_YAML}")
        print(f"Train with: uv run python recipes/roboflow/finetune/yolo/main.py --config_dir {args.config}")
        return 0

    train_cmd = [
        py,
        "recipes/roboflow/finetune/yolo/main.py",
        "--config_dir",
        args.config,
    ]
    if args.resume:
        train_cmd.extend(["--resume", args.resume])
    _run(train_cmd, "Fine-tune YOLO26 on chess-pieces-2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

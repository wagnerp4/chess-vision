import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
os.chdir(root)
sys.path.insert(0, str(root))

from recipes.roboflow.finetune.setting import (
    DEFAULT_CONFIG_RFDETR,
    DEFAULT_CONFIG_YOLO,
    prepare_run,
)
from recipes.roboflow.finetune.rfdetr.train import train_rfdetr
from recipes.roboflow.finetune.yolo.train import resume_yolo, train_yolo
from src.training.run_mode import parse_run_mode


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune detection models on Roboflow chess datasets"
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=("yolo", "rfdetr"),
        help="Vendor framework: ultralytics YOLO or RF-DETR",
    )
    parser.add_argument(
        "--config_dir",
        "--config",
        type=str,
        default=None,
        dest="config_dir",
        help="Training YAML (default: configs/roboflow/yolo/yolo.yaml or configs/roboflow/rfdetr/rfdetr.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="YOLO only: path to last.pt (not best.pt)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fast-dev",
        action="store_true",
        help="RF-DETR only: 1 epoch, batch_size=1, val-only post-train eval",
    )
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help="RF-DETR only: 1 epoch, production batch, full post-train eval",
    )
    return parser.parse_args()


def main() -> None:
    from recipes.roboflow.setting import bootstrap_repo
    from src.training.paths import load_config

    args = parse_args()
    root = bootstrap_repo()

    default = DEFAULT_CONFIG_YOLO if args.framework == "yolo" else DEFAULT_CONFIG_RFDETR
    config = load_config(args.config_dir or default, root)

    if args.framework == "yolo":
        if args.resume:
            resume_path = Path(args.resume).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            results = resume_yolo(config, resume_path, root)
        else:
            results = train_yolo(config, root)
    else:
        if args.resume:
            raise ValueError("--resume is only supported for framework=yolo")
        if args.fast_dev or args.smoke:
            run_mode = parse_run_mode(args)
        else:
            run_mode = "full"
        results = train_rfdetr(config, root, run_mode=run_mode)

    print("Training completed!")
    if args.framework == "yolo":
        print(f"Results saved to: {results.save_dir}")
    else:
        print(f"Results saved to: {results.output_dir}")
        if results.archived_checkpoint:
            print(f"Archived checkpoint: {results.archived_checkpoint}")


if __name__ == "__main__":
    main()

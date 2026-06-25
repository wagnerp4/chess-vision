from __future__ import annotations

import argparse
from pathlib import Path

from recipes.roboflow.setting import bootstrap_repo
from src.training.paths import load_config

DEFAULT_CONFIG_YOLO = "configs/roboflow/yolo/yolo.yaml"
DEFAULT_CONFIG_RFDETR = "configs/roboflow/rfdetr/rfdetr.yaml"


def parse_args(
    description: str,
    default_config: str,
    *,
    allow_resume: bool = False,
    allow_run_modes: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config_dir",
        "--config",
        type=str,
        default=default_config,
        dest="config_dir",
        help="Path to training YAML",
    )
    if allow_resume:
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Path to last.pt in the Ultralytics run dir (not best.pt)",
        )
    if allow_run_modes:
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--fast-dev",
            action="store_true",
            help="RF-DETR: 1 epoch, batch_size=1, val-only post-train eval",
        )
        mode_group.add_argument(
            "--smoke",
            action="store_true",
            help="RF-DETR: 1 epoch, production batch, full post-train eval",
        )
    return parser.parse_args()


def prepare_run(
    default_config: str,
    *,
    allow_resume: bool = False,
    allow_run_modes: bool = False,
) -> tuple[argparse.Namespace, dict, Path]:
    root = bootstrap_repo()
    args = parse_args(
        "Finetune on Roboflow-exported chess datasets",
        default_config,
        allow_resume=allow_resume,
        allow_run_modes=allow_run_modes,
    )
    config = load_config(args.config_dir, root)
    return args, config, root

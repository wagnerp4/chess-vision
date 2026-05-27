from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

from src.training.checkpoints import archive_best_checkpoint
from src.training.paths import anchor_path
from src.training.yolo_params import build_training_params


def load_yolo_model(config: dict, root: Path | None = None) -> YOLO:
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]

    pretrained_dir = anchor_path("data/checkpoints/pretrained", root)
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    if pretrained:
        pretrained_path = pretrained_dir / f"{model_name}.pt"
        root_model_path = Path(f"{model_name}.pt")

        if pretrained_path.exists():
            return YOLO(str(pretrained_path))
        if root_model_path.exists():
            shutil.move(str(root_model_path), str(pretrained_path))
            print(f"Pretrained model moved to {pretrained_path}")
            return YOLO(str(pretrained_path))
        model = YOLO(model_name)
        if root_model_path.exists():
            shutil.move(str(root_model_path), str(pretrained_path))
            print(f"Pretrained model downloaded and moved to {pretrained_path}")
        return model
    return YOLO(f"{model_name}.yaml")


def train_yolo(config: dict, root: Path | None = None):
    model = load_yolo_model(config, root)
    results = model.train(**build_training_params(config, root))

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        archive_best_checkpoint(best_model_path, config, root)
    return results


def resume_yolo(config: dict, resume_path: Path, root: Path | None = None):
    model = YOLO(str(resume_path))
    params = build_training_params(config, root)
    params["resume"] = True
    results = model.train(**params)
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        archive_best_checkpoint(best_model_path, config, root)
    return results

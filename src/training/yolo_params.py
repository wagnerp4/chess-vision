from __future__ import annotations

from pathlib import Path

from src.training.optimizers import yolo_optimizer_kwargs
from src.training.paths import anchor_path, resolve_device


def resolve_dataset_yaml(config: dict, root: Path | None = None) -> Path:
    dataset_yaml = anchor_path(config["data"]["dataset_path"], root)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    return dataset_yaml


def build_training_params(config: dict, root: Path | None = None) -> dict:
    dataset_yaml = resolve_dataset_yaml(config, root)
    training = config["training"]
    device = resolve_device(training["device"])
    aug_params = training["augmentation"]

    params = {
        "data": str(dataset_yaml),
        "epochs": training["epochs"],
        "patience": training["patience"],
        "device": device,
        "project": str(anchor_path(training["project"], root)),
        "name": training["name"],
        "batch": config["data"]["batch_size"],
        "workers": config["data"]["num_workers"],
        "imgsz": config["model"]["input_size"],
        "plots": training.get("plots", False),
        "amp": training.get("amp", False),
        **yolo_optimizer_kwargs(training["optimizer"]),
        "hsv_h": aug_params["hsv_h"],
        "hsv_s": aug_params["hsv_s"],
        "hsv_v": aug_params["hsv_v"],
        "degrees": aug_params["degrees"],
        "translate": aug_params["translate"],
        "scale": aug_params["scale"],
        "shear": aug_params["shear"],
        "perspective": aug_params["perspective"],
        "flipud": aug_params["flipud"],
        "fliplr": aug_params["fliplr"],
        "mosaic": aug_params["mosaic"],
        "mixup": aug_params["mixup"],
    }
    print(f"Training device: {device}  plots={params['plots']}  amp={params['amp']}")
    return params

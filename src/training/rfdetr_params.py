from __future__ import annotations

from pathlib import Path

from src.training.optimizers import rfdetr_optimizer_kwargs
from src.training.paths import anchor_path


def resolve_dataset_yaml(config: dict, root: Path | None = None) -> Path:
    dataset_path = anchor_path(config["data"]["dataset_path"], root)
    if dataset_path.is_dir():
        dataset_yaml = dataset_path / "dataset.yaml"
    elif dataset_path.suffix in (".yaml", ".yml"):
        dataset_yaml = dataset_path
    else:
        dataset_yaml = dataset_path / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    return dataset_yaml


def build_training_params(config: dict, root: Path | None = None) -> dict:
    dataset_yaml = resolve_dataset_yaml(config, root)
    training = config["training"]
    return {
        "data": str(dataset_yaml),
        "epochs": training["epochs"],
        "batch_size": config["data"]["batch_size"],
        "device": training["device"],
        "output_dir": training["save_dir"],
        **rfdetr_optimizer_kwargs(training["optimizer"]),
    }

from __future__ import annotations

from typing import Any


def yolo_optimizer_kwargs(optimizer_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "lr0": optimizer_cfg["lr0"],
        "lrf": optimizer_cfg["lrf"],
        "momentum": optimizer_cfg["momentum"],
        "weight_decay": optimizer_cfg["weight_decay"],
        "warmup_epochs": optimizer_cfg["warmup_epochs"],
        "warmup_momentum": optimizer_cfg["warmup_momentum"],
        "warmup_bias_lr": optimizer_cfg["warmup_bias_lr"],
    }


def rfdetr_optimizer_kwargs(optimizer_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": optimizer_cfg["lr"],
        "weight_decay": optimizer_cfg["weight_decay"],
        "warmup_epochs": optimizer_cfg["warmup_epochs"],
    }

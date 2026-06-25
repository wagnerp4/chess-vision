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
    kwargs: dict[str, Any] = {
        "lr": optimizer_cfg["lr"],
        "weight_decay": optimizer_cfg["weight_decay"],
        "warmup_epochs": optimizer_cfg["warmup_epochs"],
    }
    if "lr_encoder" in optimizer_cfg:
        kwargs["lr_encoder"] = optimizer_cfg["lr_encoder"]
    if "lr_scheduler" in optimizer_cfg:
        kwargs["lr_scheduler"] = optimizer_cfg["lr_scheduler"]
    return kwargs

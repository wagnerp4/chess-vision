from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.training.paths import anchor_path
from src.training.run_mode import RunMode, effective_run_name

RFDETR_BEST_CHECKPOINT = "checkpoint_best_total.pth"
RFDETR_ARCHIVE_STEM = "rfdetr"
RFDETR_PORTABLE_KEYS = (
    "model",
    "args",
    "epoch",
    "state_dict",
    "global_step",
    "model_name",
    "model_config",
    "rfdetr_version",
)


def find_best_checkpoint(output_dir: Path) -> Path | None:
    for name in (
        RFDETR_BEST_CHECKPOINT,
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
    ):
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


def resolve_rfdetr_source(source: str | Path) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    if path.is_dir():
        best = find_best_checkpoint(path)
        if best is not None:
            return best
        ckpts = sorted(path.glob("checkpoint_*.ckpt"))
        if ckpts:
            return ckpts[-1]
        raise FileNotFoundError(f"No RF-DETR checkpoint found under {path}")
    raise FileNotFoundError(f"RF-DETR source not found: {path}")


def build_portable_checkpoint(source_path: Path) -> dict[str, Any]:
    payload = torch.load(str(source_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {source_path}")

    if "args" in payload and "state_dict" in payload:
        return {key: payload[key] for key in RFDETR_PORTABLE_KEYS if key in payload}

    if "state_dict" in payload and str(source_path).endswith(".ckpt"):
        stripped = {
            key.replace("model.", "", 1): value
            for key, value in payload["state_dict"].items()
            if key.startswith("model.")
        }
        scaffold = find_best_checkpoint(source_path.parent)
        if scaffold is None:
            raise FileNotFoundError(
                f"Lightning checkpoint {source_path} needs a sibling "
                "checkpoint_best_ema.pth (or regular/total) for metadata."
            )
        scaffold_payload = torch.load(str(scaffold), map_location="cpu", weights_only=False)
        portable = {
            key: scaffold_payload[key]
            for key in RFDETR_PORTABLE_KEYS
            if key in scaffold_payload
        }
        portable["state_dict"] = stripped
        portable["epoch"] = payload.get("epoch", scaffold_payload.get("epoch"))
        portable["global_step"] = payload.get("global_step", scaffold_payload.get("global_step"))
        return portable

    raise ValueError(
        f"Cannot build portable RF-DETR checkpoint from {source_path}. "
        "Expected .pth with args/state_dict or .ckpt with sibling scaffold."
    )


def export_rfdetr_checkpoint(
    source: str | Path,
    output_path: str | Path | None = None,
    root: Path | None = None,
    date: str | None = None,
) -> Path:
    source_path = resolve_rfdetr_source(source)
    best_dir = anchor_path("data/checkpoints/best", root)
    best_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        dated = date or datetime.now().strftime("%Y%m%d")
        output_path = best_dir / f"best_{dated}_{RFDETR_ARCHIVE_STEM}.pt"
    else:
        output_path = Path(output_path)
        if root is not None and not output_path.is_absolute():
            output_path = root / output_path

    portable = build_portable_checkpoint(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"RF-DETR portable checkpoint exported to {output_path} "
        f"({size_mb:.2f} MB) from {source_path.name}"
    )
    return output_path


def archive_rfdetr_checkpoint(
    checkpoint_path: Path,
    config: dict,
    root: Path | None = None,
    run_mode: RunMode = "full",
) -> Path | None:
    save_dir = anchor_path(config["training"]["save_dir"], root)
    run_name = effective_run_name(config, run_mode)
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    local_copy = run_dir / RFDETR_BEST_CHECKPOINT
    if checkpoint_path.resolve() != local_copy.resolve():
        shutil.copy2(checkpoint_path, local_copy)
        print(f"Best RF-DETR checkpoint copied to {local_copy}")
    else:
        print(f"Best RF-DETR checkpoint at {local_copy}")

    dated = datetime.now().strftime("%Y%m%d")
    suffix = ""
    if run_mode == "fast_dev":
        suffix = "_fastdev"
    elif run_mode == "smoke":
        suffix = "_smoke"
    archive_name = f"best_{dated}_{RFDETR_ARCHIVE_STEM}{suffix}.pt"
    archived = anchor_path("data/checkpoints/best", root) / archive_name
    return export_rfdetr_checkpoint(checkpoint_path, output_path=archived, root=root, date=dated)

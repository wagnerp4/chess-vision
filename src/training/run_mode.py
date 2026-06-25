from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from src.data.class_mapping import CANONICAL_CLASSES
from src.training.paths import anchor_path

RunMode = Literal["full", "fast_dev", "smoke"]


def parse_run_mode(args: Any) -> RunMode:
    if getattr(args, "fast_dev", False):
        return "fast_dev"
    if getattr(args, "smoke", False):
        return "smoke"
    return "full"


def run_name_suffix(mode: RunMode) -> str:
    if mode == "fast_dev":
        return "_fastdev"
    if mode == "smoke":
        return "_smoke"
    return ""


def apply_run_mode(config: dict, mode: RunMode) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "class_names": list(CANONICAL_CLASSES),
        "log_per_class_metrics": True,
        "eval_interval": 1,
        "progress_bar": "tqdm",
    }
    if mode == "full":
        training = config.get("training", {})
        overrides["skip_best_epochs"] = training.get("skip_best_epochs", 3)
        overrides["early_stopping"] = True
        overrides["early_stopping_patience"] = training.get("patience", 20)
        return overrides

    overrides["epochs"] = 1
    overrides["early_stopping"] = False
    overrides["skip_best_epochs"] = 0
    overrides["warmup_epochs"] = 0

    if mode == "fast_dev":
        overrides["batch_size"] = 1
        overrides["grad_accum_steps"] = 1

    return overrides


def effective_run_name(config: dict, mode: RunMode) -> str:
    base = config["training"]["name"]
    return f"{base}{run_name_suffix(mode)}"


def filter_post_train_entries(
    config: dict,
    mode: RunMode,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    evaluation = config.get("evaluation") or {}
    entries = list(evaluation.get("post_train") or [])
    if mode != "fast_dev":
        return entries
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        split = entry.get("split", "val")
        if split in ("val", "valid"):
            filtered.append(entry)
    return filtered


def check_val_class_coverage(config: dict, root: Path | None = None) -> None:
    dataset_path = anchor_path(config["data"]["dataset_path"], root)
    if dataset_path.is_file():
        dataset_root = dataset_path.parent
    else:
        dataset_root = dataset_path
    labels_dir = dataset_root / "valid" / "labels"
    if not labels_dir.is_dir():
        labels_dir = dataset_root / "val" / "labels"
    if not labels_dir.is_dir():
        print(f"Warning: val labels not found under {dataset_root}, skipping class coverage check")
        return
    seen: set[int] = set()
    for label_path in labels_dir.glob("*.txt"):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    seen.add(int(parts[0]))
    missing = set(range(len(CANONICAL_CLASSES))) - seen
    if missing:
        names = [CANONICAL_CLASSES[i] for i in sorted(missing)]
        print(f"Warning: val split missing GT for class IDs {sorted(missing)}: {names}")

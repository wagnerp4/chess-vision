from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.optimizers import rfdetr_optimizer_kwargs
from src.training.paths import anchor_path, resolve_rfdetr_device
from src.training.run_mode import RunMode, apply_run_mode, effective_run_name


def resolve_dataset_dir(config: dict, root: Path | None = None) -> Path:
    dataset_path = anchor_path(config["data"]["dataset_path"], root)
    if dataset_path.is_dir():
        return dataset_path
    if dataset_path.suffix in (".yaml", ".yml"):
        return dataset_path.parent
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    return dataset_path


def build_training_params(
    config: dict,
    root: Path | None = None,
    run_mode: RunMode = "full",
) -> dict[str, Any]:
    dataset_dir = resolve_dataset_dir(config, root)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        dataset_yaml = dataset_dir / "dataset.yaml"
        if dataset_yaml.exists():
            raise FileNotFoundError(
                f"RF-DETR requires data.yaml under {dataset_dir}. "
                f"Found dataset.yaml only; re-run normalize_dataset or copy to data.yaml."
            )
        raise FileNotFoundError(f"Dataset YAML not found under {dataset_dir}")

    training = config["training"]
    data_cfg = config["data"]
    device = resolve_rfdetr_device(training["device"])
    run_name = effective_run_name(config, run_mode)
    output_dir = anchor_path(training["save_dir"], root) / run_name

    params: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "epochs": training["epochs"],
        "batch_size": data_cfg["batch_size"],
        "grad_accum_steps": data_cfg.get("grad_accum_steps", 4),
        "device": device,
        "output_dir": str(output_dir),
        "num_workers": data_cfg.get("num_workers", 2),
        "run": run_name,
        **rfdetr_optimizer_kwargs(training["optimizer"]),
    }

    input_size = config.get("model", {}).get("input_size")
    if input_size is not None:
        params["resolution"] = int(input_size)

    params.update(apply_run_mode(config, run_mode))

    training_cfg = config.get("training", {})
    if training_cfg.get("progress_bar"):
        params["progress_bar"] = training_cfg["progress_bar"]

    print(
        f"RF-DETR training: device={device}  run={run_name}  mode={run_mode}  "
        f"batch={params['batch_size']}  grad_accum={params['grad_accum_steps']}  "
        f"epochs={params['epochs']}"
    )
    return params

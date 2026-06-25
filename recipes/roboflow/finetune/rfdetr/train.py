from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.backends.rfdetr import build_rfdetr
from src.training.post_train_eval import run_post_train_eval, write_benchmark_json
from src.training.rfdetr_checkpoints import archive_rfdetr_checkpoint, find_best_checkpoint
from src.training.rfdetr_params import build_training_params
from src.training.run_mode import RunMode, check_val_class_coverage


@dataclass
class RFDETRTrainResult:
    output_dir: Path
    checkpoint_path: Path | None
    archived_checkpoint: Path | None


def _trainer_kwargs_for_mode(config: dict, run_mode: RunMode) -> dict[str, Any]:
    training = config.get("training", {})
    kwargs: dict[str, Any] = {
        "log_every_n_steps": int(training.get("log_every_n_steps", 10)),
    }
    if run_mode == "fast_dev":
        max_batches = int(training.get("fast_dev_max_batches", 64))
        kwargs["limit_train_batches"] = max_batches
        print(
            f"fast-dev: tqdm progress on train batches (max {max_batches}); "
            f"full val COCO eval still runs"
        )
    return kwargs


def _run_train(model: Any, params: dict[str, Any], trainer_kwargs: dict[str, Any]) -> None:
    if not trainer_kwargs:
        model.train(**params)
        return

    import rfdetr.training as rfdetr_training

    original_build = rfdetr_training.build_trainer

    def patched_build(tc: Any, mc: Any, **kwargs: Any) -> Any:
        merged = {**kwargs, **trainer_kwargs}
        return original_build(tc, mc, **merged)

    rfdetr_training.build_trainer = patched_build
    try:
        model.train(**params)
    finally:
        rfdetr_training.build_trainer = original_build


def train_rfdetr(
    config: dict,
    root: Path | None = None,
    run_mode: RunMode = "full",
) -> RFDETRTrainResult:
    if run_mode in ("fast_dev", "smoke"):
        check_val_class_coverage(config, root)

    model_size = config.get("model", {}).get("size", "large").lower()
    model = build_rfdetr(model_size)
    params = build_training_params(config, root, run_mode=run_mode)
    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = _trainer_kwargs_for_mode(config, run_mode)
    _run_train(model, params, trainer_kwargs)

    best_path = find_best_checkpoint(output_dir)
    archived: Path | None = None
    if best_path is not None:
        archived = archive_rfdetr_checkpoint(best_path, config, root, run_mode=run_mode)
    else:
        print(f"Warning: no RF-DETR best checkpoint found under {output_dir}")

    eval_checkpoint = archived or best_path
    if eval_checkpoint is not None and config.get("evaluation", {}).get("post_train"):
        manifest = run_post_train_eval(
            config,
            eval_checkpoint,
            model_size,
            root,
            run_mode=run_mode,
        )
        write_benchmark_json(manifest, config, root, run_mode=run_mode)

    return RFDETRTrainResult(
        output_dir=output_dir,
        checkpoint_path=best_path,
        archived_checkpoint=archived,
    )

from __future__ import annotations

import shutil
from pathlib import Path

from src.models.backends.rfdetr import build_rfdetr
from src.training.paths import anchor_path
from src.training.rfdetr_params import build_training_params


def train_rfdetr(config: dict, root: Path | None = None):
    model_size = config.get("model", {}).get("size", "base").lower()
    model = build_rfdetr(model_size)
    results = model.train(**build_training_params(config, root))

    save_dir = anchor_path(config["training"]["save_dir"], root)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(results.save_dir) / "best.pt"
    if best_model_path.exists():
        shutil.copy(best_model_path, save_dir / "best.pt")
        print(f"Best model saved to {save_dir / 'best.pt'}")

    return results

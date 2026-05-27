from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from src.training.paths import anchor_path


def archive_best_checkpoint(best_model_path: Path, config: dict, root: Path | None = None) -> None:
    save_dir = anchor_path(config["training"]["save_dir"], root)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, save_dir / "best.pt")
    print(f"Best model saved to {save_dir / 'best.pt'}")

    best_checkpoints_dir = anchor_path("data/checkpoints/best", root)
    best_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    max_size_mb = 50.0
    run_name = config["training"]["name"]

    if file_size_mb <= max_size_mb:
        dated = datetime.now().strftime("%Y%m%d")
        versioned_path = best_checkpoints_dir / f"best_{dated}_{run_name}.pt"
        shutil.copy(best_model_path, versioned_path)
        print(f"Best checkpoint archived to {versioned_path} ({file_size_mb:.2f} MB)")
    else:
        print(
            f"Best checkpoint ({file_size_mb:.2f} MB) exceeds size limit ({max_size_mb} MB), not archiving"
        )

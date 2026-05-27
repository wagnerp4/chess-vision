# data/

Local artifacts and datasets. The repo-root `data/` tree is **gitignored** except for this README, `data/roboflow_presets.yaml`, and **`data/checkpoints/best/`** (archived `.pt` + `benchmark_*.json`). `pretrained/`, `yolo/`, and `oak/` stay ignored. Re-create other subdirs with download or training scripts.

## Layout

| Path | Purpose |
|------|---------|
| `roboflow/` | Raw Roboflow exports (`scripts/download_roboflow.py`) |
| `processed/` | Normalized YOLO `dataset.yaml` trees (`scripts/normalize_dataset.py`) |
| `kaggle/` | Legacy Kaggle downloads |
| `runs/` | Ultralytics run outputs (`training.project` in config) |
| `exps/` | Older experiment logs (optional) |
| `checkpoints/` | Pretrained weights, archived bests, latest `best.pt` — see `checkpoints/README.md` |

## Roboflow presets

Download preset definitions live in `data/roboflow_presets.yaml` (not under `configs/`).

## Training configs

Point `data.dataset_path` in `configs/yolo.yaml` or `configs/rfdetr.yaml` at a processed dataset, e.g. `data/processed/chess-pieces-2/dataset.yaml`.

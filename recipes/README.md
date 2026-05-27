# recipes/

Grouped by **dataset vendor**, not model architecture (same idea as `recipes/datasets/` in SSL4SED). Under each vendor, training stages such as `finetune/` host framework-specific entrypoints (YOLO, RF-DETR, …).

## Layout

| Vendor | Stage | Framework | Entry | Default config |
|--------|-------|-----------|--------|----------------|
| `roboflow/` | `finetune/` | YOLO (Ultralytics) | `recipes/roboflow/finetune/yolo/main.py` | `configs/roboflow/yolo/yolo.yaml` |
| `roboflow/` | `finetune/` | RF-DETR | `recipes/roboflow/finetune/rfdetr/main.py` | `configs/roboflow/rfdetr/rfdetr.yaml` |
| `roboflow/` | `finetune/` | either | `recipes/roboflow/finetune/main.py` | `--framework yolo\|rfdetr` |

Shared wiring: `recipes/roboflow/setting.py` (repo bootstrap), `recipes/roboflow/finetune/setting.py` (CLI + config load). Training loops: `finetune/<framework>/train.py`. Optimizer kwargs and paths: `src/training/`. Model backends: `src/models/backends/`. Metrics: `src/metrics/tasks/obj_det/`. Downloads: `src/utils/download/`.

## Run

```bash
cd vision

uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo/yolo.yaml
uv run python recipes/roboflow/finetune/rfdetr/main.py --config_dir configs/roboflow/rfdetr/rfdetr.yaml

uv run python recipes/roboflow/finetune/main.py --framework yolo --config_dir configs/roboflow/yolo/yolo.yaml
uv run python recipes/roboflow/finetune/main.py --framework yolo --config_dir configs/roboflow/yolo/yolo.yaml \
  --resume path/to/weights/last.pt
```

Dataset download presets: `data/roboflow_presets.yaml`. Processed trees: `data/processed/` (see `data/README.md`).

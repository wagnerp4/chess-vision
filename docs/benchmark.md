# Chess Pieces 2 — detection benchmark

Comparison on the **Roboflow Chess Pieces 2** processed dataset (`data/processed/chess-pieces-2/dataset.yaml`), split **`val`** (910 images). Ultralytics `model.val()` with `conf=None`, `iou=0.7` (NMS). End-of-train logs use the same split and `conf=None` unless you changed training args.

**Official Roboflow 3.0** scores (Universe model card, v5) are the reference row.

| Model | mAP@50 | Precision | Recall |
|-------|--------|-----------|--------|
| Roboflow 3.0 (chess-pieces-2 v5, Universe) | 96.2% | 96.8% | 94.4% |
| best_20260110_122310 (OAK export, not 12-class) | 1.5% | 2.2% | 40.5% |
| best_20260515_yolo_chess_pieces_2 (YOLOv8n) | 95.0% | 95.2% | 92.8% |
| best_20260527_yolo26n_chess_pieces_2 (YOLO26n) | 94.8% | 93.9% | 91.2% |

## Per-class metrics (local checkpoints)

Class-wise scores are written next to each checkpoint run:

- `data/runs/detect/yolo26n_chess_pieces_2-2/benchmark_val.json` (YOLO26n run)
- `data/checkpoints/best/benchmark_<checkpoint_stem>.json` for checkpoints without a linked run dir

Regenerate this table:

```bash
cd vision
uv run python scripts/benchmark_yolo_checkpoints.py --split val
```

## Notes

- `infer_yolo.py` is for **single-image** inference and visualization only.
- Use `scripts/evaluate_models.py` or this script for **mAP@50 / P / R**.
- **Split does not change** between train-end val and this table: both use Roboflow `valid` → YAML key `val` (910 images, 57 batches at batch 16).
- **92.3% vs 94.8%:** the first benchmark run used `conf=0.25` (from `configs/yolo.yaml` inference defaults). Training val uses `conf=None` (low threshold for mAP). Re-run with `--conf none` to match the training log.
- `iou=0.7` in training is **NMS** IoU, not the 0.5 used inside mAP@50 box matching.
- Roboflow official numbers may use their own conf/NMS. Treat gaps as indicative.
- `best_20260110_122310` scores ~0% on this 12-class layout (likely OAK export, not comparable).

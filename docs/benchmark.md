# Detector benchmark leaderboard

Unified **predict-then-score** evaluation. Every backend runs inference on the same frozen split, then metrics are computed by shared scorers (no backend-specific `val()`).

## Protocol

- Manifest: `configs/benchmark.yaml` (datasets, splits, models, thresholds)
- Primary metric: **COCO mAP@50** (`metrics.coco.mAP50`)
- Secondary: COCO mAP@50:95, legacy 11-point per-image AP (`metrics.legacy_ap.mAP50`)
- Eval config: `conf=0.25`, `iou_match=0.5`, `iou_nms=0.45`, `imgsz=640`

Regenerate:

```bash
cd Vision/Personal/chess-vision
uv run python scripts/evaluate_models.py --manifest configs/benchmark.yaml
```

Results JSON: `results/benchmark/leaderboard.json` plus per-cell files under `results/benchmark/<model_id>/`.

## Notes

- `chess_pieces2_test` is the official Roboflow held-out split (16 images, high variance).
- `chess_pieces2_val` is the stable in-domain split (910 images).
- `neuroTUM_test` uses pseudo-GT from Roboflow teacher labels (810 frames, domain shift).
- Add a new detector: implement `DetectorBackend` under `src/evaluation/backends/` and register in `registry.py`.
- **Live OAK benchmark:** see [commands.md](commands.md) §6 (`scripts/live_benchmark.py`, `scripts/chess_pipeline.py`).

## chess_pieces2_test (label_type=human)

_Chess Pieces 2 official test (16 images)_

| Model | COCO mAP@50 | COCO mAP@50:95 | Legacy mAP@50 | Images |
|-------|-------------|----------------|---------------|--------|
| RF-DETR large ep29 | 97.4% | 82.2% | 96.0% | 16 |
| YOLOv8n chess-pieces-2 | 95.4% | 79.3% | 93.0% | 16 |
| YOLO26n chess-pieces-2 | 91.7% | 76.9% | 90.7% | 16 |

## chess_pieces2_val (label_type=human)

_Chess Pieces 2 val (910 images)_

| Model | COCO mAP@50 | COCO mAP@50:95 | Legacy mAP@50 | Images |
|-------|-------------|----------------|---------------|--------|
| RF-DETR large ep29 | 94.2% | 65.9% | 95.9% | 910 |
| YOLOv8n chess-pieces-2 | 92.3% | 64.7% | 93.7% | 910 |
| YOLO26n chess-pieces-2 | 91.8% | 63.7% | 92.6% | 910 |

## neuroTUM_test (label_type=pseudo)

_neuroTUM domain shift (810 frames, pseudo-GT)_

| Model | COCO mAP@50 | COCO mAP@50:95 | Legacy mAP@50 | Images |
|-------|-------------|----------------|---------------|--------|
| YOLOv8n chess-pieces-2 | 4.2% | 1.8% | 5.0% | 810 |
| RF-DETR large ep29 | 4.0% | 1.6% | 4.4% | 810 |
| YOLO26n chess-pieces-2 | 3.9% | 1.8% | 3.8% | 810 |

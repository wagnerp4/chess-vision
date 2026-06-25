# Command reference

All commands assume the repo root:

```bash
cd Vision/Personal/chess-vision
```

Related docs: [datasets.md](datasets.md), [benchmark.md](benchmark.md), [models.md](models.md), [grid_detector.md](grid_detector.md).

---

## 1. Data

Download Roboflow Chess Pieces 2 and normalize class names:

```bash
uv run python scripts/download_roboflow.py --preset chess-pieces-2
uv run python scripts/normalize_dataset.py --input data/roboflow/chess-pieces-2
```

List presets:

```bash
uv run python scripts/download_roboflow.py --list-presets
```

Build pseudo-label test set from video (neuroTUM):

```bash
uv run python scripts/build_video_pseudo_dataset.py
```

Legacy Kaggle download:

```bash
uv run python scripts/download_dataset.py
```

---

## 2. Training

Configs: `configs/roboflow/yolo/yolo.yaml`, `configs/roboflow/yolo26/yolo26.yaml`, `configs/roboflow/rfdetr/rfdetr.yaml`.

### YOLO (Ultralytics)

```bash
uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo/yolo.yaml
```

YOLO26:

```bash
uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo26/yolo26.yaml
```

Resume (use `last.pt`, not `best.pt`):

```bash
uv run python recipes/roboflow/finetune/yolo/main.py \
  --config_dir configs/roboflow/yolo/yolo.yaml \
  --resume path/to/weights/last.pt
```

Unified entry (either framework):

```bash
uv run python recipes/roboflow/finetune/main.py --framework yolo --config_dir configs/roboflow/yolo/yolo.yaml
```

On Apple Silicon, set `training.device: "mps"` in the YAML. On NVIDIA Linux, install CUDA wheels first (see README).

Archived weights land in `data/checkpoints/best/best_YYYYMMDD_<run_name>.pt` (YOLO only if ≤ 50 MB).

### RF-DETR

Full training:

```bash
uv run python recipes/roboflow/finetune/rfdetr/main.py --config_dir configs/roboflow/rfdetr/rfdetr.yaml
```

Smoke / fast-dev (1 epoch, post-train eval):

```bash
uv run python recipes/roboflow/finetune/main.py --framework rfdetr --config_dir configs/roboflow/rfdetr/rfdetr.yaml --fast-dev
uv run python recipes/roboflow/finetune/main.py --framework rfdetr --config_dir configs/roboflow/rfdetr/rfdetr.yaml --smoke
```

Training artifacts go to `data/checkpoints/rfdetr/` (gitignored). Export the portable checkpoint after training (see §3).

### Pretrained download

```bash
uv run python scripts/download_yolo_pretrained.py
```

---

## 3. Checkpoint export

### RF-DETR → portable `best_YYYYMMDD_rfdetr.pt`

RF-DETR bundles are ~130 MB and are **not committed** (GitHub 100 MB limit). Export locally:

```bash
uv run python scripts/export_rfdetr_checkpoint.py
```

Explicit source and date:

```bash
uv run python scripts/export_rfdetr_checkpoint.py \
  --source data/checkpoints/rfdetr/rfdetr_large_chess_pieces_2 \
  --date 20260625
```

Output: `data/checkpoints/best/best_20260625_rfdetr.pt`

### YOLO → OAK `.blob`

```bash
uv run python scripts/convert_to_oak.py --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt
```

On-device blob inference (YOLO only, no grid warp):

```bash
uv run python scripts/run_chess_detection_oak.py --model data/checkpoints/oak/best_20260515_yolo_chess_pieces_2.blob
```

---

## 4. Offline evaluation

Unified backend-agnostic benchmark (COCO + legacy AP on frozen splits). Manifest: `configs/benchmark.yaml`.

Full matrix (YOLOv8, YOLO26, RF-DETR × chess-pieces-2 + neuroTUM):

```bash
uv run python scripts/evaluate_models.py --manifest configs/benchmark.yaml
```

Single model × scenario:

```bash
uv run python scripts/evaluate_models.py \
  --manifest configs/benchmark.yaml \
  --model-id yolov8n_cp2 \
  --scenario-id chess_pieces2_val
```

Ad-hoc one-off (legacy flags):

```bash
uv run python scripts/evaluate_models.py \
  --data data/processed/chess-pieces-2/dataset.yaml \
  --split val \
  --yolo-model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt \
  --conf 0.25 \
  --output results/evaluation.json
```

```bash
uv run python scripts/evaluate_models.py \
  --data data/processed/chess-pieces-2/dataset.yaml \
  --split val \
  --rfdetr-size large \
  --rfdetr-model data/checkpoints/best/best_20260625_rfdetr.pt \
  --output results/evaluation_rfdetr.json
```

YOLO-only val table (updates `docs/benchmark.md`):

```bash
uv run python scripts/benchmark_yolo_checkpoints.py --split val
```

Video pseudo-GT subset (neuroTUM):

```bash
uv run python scripts/run_video_subset_inference.py \
  --model data/checkpoints/best/best_20260527_yolo26n_chess_pieces_2.pt \
  --sample-size 50 --metrics
```

Single hero image:

```bash
uv run python scripts/run_benchmark_image.py --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt
```

### Evaluation flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--manifest` | — | Run full benchmark from YAML |
| `--model-id` / `--scenario-id` | — | Single cell from manifest |
| `--data` | chess-pieces-2 | Dataset yaml or directory |
| `--split` | test | `train`, `val`, `test` |
| `--conf` | 0.25 | Detection confidence |
| `--iou-match` | 0.5 | Box matching IoU for mAP |
| `--iou-nms` | 0.45 | NMS IoU (ultralytics) |
| `--output` | `results/benchmark` | JSON output path or directory |

Results: `results/benchmark/leaderboard.json`, leaderboard table in [benchmark.md](benchmark.md).

---

## 5. Single-image inference

YOLO:

```bash
uv run python src/inference/infer_yolo.py \
  --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt \
  --image data/test_images/your_image.jpg \
  --output outputs/annotated.jpg \
  --conf 0.25
```

RF-DETR:

```bash
uv run python src/inference/infer_rfdetr.py \
  --model data/checkpoints/best/best_20260625_rfdetr.pt \
  --model-size large \
  --image data/test_images/your_image.jpg \
  --output outputs/annotated.jpg \
  --conf 0.25
```

Grid detection only (no piece model):

```bash
uv run python scripts/test_grid_detection.py --image path/to/image.jpg
uv run python scripts/grid_camera.py
```

---

## 6. Real-time OAK-D Lite

Pipeline: **camera → LK grid track → warp 800×800 → piece detector**. Backends: `ultralytics` (YOLO `.pt`) or `rfdetr` (portable `.pt`).

Requires OAK-D Lite connected via USB 3.0 and `depthai` installed.

### Interactive (OpenCV windows)

YOLO (default `piece_every=3`):

```bash
uv run python scripts/chess_pipeline.py --backend ultralytics --piece-every 3
```

RF-DETR (slower — use higher `piece_every`):

```bash
uv run python scripts/chess_pipeline.py \
  --backend rfdetr \
  --weights data/checkpoints/best/best_20260625_rfdetr.pt \
  --piece-every 8
```

With explicit weights:

```bash
uv run python scripts/chess_pipeline.py \
  --backend ultralytics \
  --weights data/checkpoints/best/best_20260527_yolo26n_chess_pieces_2.pt \
  --conf 0.25 \
  --piece-every 3
```

Keyboard: `q` quit, `r` reset grid tracking, `s` save snapshot to `outputs/live_snapshots/`.

### Timed live benchmark (latency JSON)

Headless, writes timing report:

```bash
uv run python scripts/live_benchmark.py \
  --backend ultralytics \
  --piece-every 3 \
  --duration 30 \
  --output results/live_benchmark_yolo.json
```

```bash
uv run python scripts/live_benchmark.py \
  --backend rfdetr \
  --weights data/checkpoints/best/best_20260625_rfdetr.pt \
  --piece-every 8 \
  --duration 30 \
  --show \
  --output results/live_benchmark_rfdetr.json
```

### Real-time flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--backend` | ultralytics | `ultralytics` or `rfdetr` |
| `--weights` / `--model` | YOLO best | Checkpoint path |
| `--rfdetr-size` | large | RF-DETR variant |
| `--piece-every` | 3 (YOLO), 6 (RF-DETR) | Update piece bboxes every N camera frames |
| `--grid-every` | 2 | Full grid detect every N frames while LK tracking |
| `--redetect-every` | 60 | Force grid re-anchor every N frames |
| `--conf` | 0.25 | Detection confidence |
| `--duration` | 0 | Auto-stop after N seconds (0 = until `q`) |
| `--show` | off (benchmark) / on (pipeline) | OpenCV windows |
| `--camera-width` / `--camera-height` | 1280×720 | OAK RGB resolution |

At ~27 fps, `piece_every=3` refreshes boxes ~9×/s. RF-DETR at `piece_every=8` → ~3.4×/s. Stale boxes are held between inference frames (usually imperceptible).

Implementation: `src/inference/realtime.py`. Live metrics: `src/evaluation/live_runner.py`.

### UDP board stream

Headless stream with optional robot framing:

```bash
uv run python scripts/udp_streamer.py \
  --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt \
  --conf 0.25 \
  --piece-every 3 \
  --udp-port 9100

uv run python scripts/udp_streamer.py --show --robot
```

Listen: `nc -u -l 127.0.0.1 9100`

---

## 7. Checkpoints quick reference

| File | Backend | In git? |
|------|---------|---------|
| `data/checkpoints/best/best_*_yolo*.pt` | Ultralytics YOLO | yes (~6 MB) |
| `data/checkpoints/best/best_*_rfdetr.pt` | RF-DETR portable | no (export locally) |
| `data/checkpoints/rfdetr/` | RF-DETR training runs | no (gitignored) |
| `data/checkpoints/oak/*.blob` | OAK on-device YOLO | optional |

Benchmark manifest weights: `configs/benchmark.yaml`.

---

## 8. Utility scripts

| Script | Purpose |
|--------|---------|
| `scripts/create_demo_video.py` | Annotated demo video from dataset |
| `scripts/plot_nt_dataset_debug.py` | Visualize neuroTUM pseudo labels |
| `scripts/run_yolo26_chess_pieces_2.py` | YOLO26 training helper |
| `scripts/chess_pipeline.py` | Live OAK pipeline (interactive) |
| `scripts/live_benchmark.py` | Live OAK latency benchmark |
| `scripts/evaluate_models.py` | Offline unified evaluation |
| `scripts/export_rfdetr_checkpoint.py` | RF-DETR → `best_*_rfdetr.pt` |
| `scripts/convert_to_oak.py` | YOLO `.pt` → `.blob` |

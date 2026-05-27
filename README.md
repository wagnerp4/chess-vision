# Chess-Vision Project

## Dataset install (Roboflow)

Downloads are scripted via the Roboflow API (no browser after setup). Large archives stay under `data/roboflow/` and `data/processed/` (gitignored).

### 1. API key

Create a free account at [Roboflow](https://app.roboflow.com/), copy your key from [Settings → API](https://app.roboflow.com/settings/api), then set it in the repo root `.env`:

```bash
# .env (repo root, gitignored)
ROBOFLOW_API_KEY="your_key_here"
```

Or export in the shell: `export ROBOFLOW_API_KEY="your_key_here"`.

### 2. Download and normalize

Run from the `vision/` directory:

```bash
cd vision

# Primary set (Chess Pieces 2, v5 @ 640×640)
uv run python scripts/download_roboflow.py --preset chess-pieces-2

# Optional smaller regression set (Public Domain, tripod + aug)
uv run python scripts/download_roboflow.py --preset chess-full-raw
uv run python scripts/download_roboflow.py --preset chess-full-aug

# Canonical 12-class layout → data/processed/<preset>/dataset.yaml
uv run python scripts/normalize_dataset.py --input data/roboflow/chess-pieces-2
```

List presets and overrides: `uv run python scripts/download_roboflow.py --list-presets`  
Presets live in [data/roboflow_presets.yaml](data/roboflow_presets.yaml).

**Kaggle (legacy):** `uv run python scripts/download_dataset.py` (requires `~/.kaggle/kaggle.json`).

### 3. Chess Pieces 2 statistics (v5)

Verified after download on 2026-05-15:

| | |
|---|---|
| Source | `fhv/chess-pieces-2-6l8qq` version **5** |
| License | CC BY 4.0 |
| Resolution | 640×640 (fit, white edges) |
| Classes | 12 (`white-pawn` … `black-king`) |
| **Total images** | **6,588** |

| Split | Images | Share |
|-------|--------|-------|
| train | 5,662 | 86% |
| valid | 910 | 14% |
| test | 16 | under 1% |

Training configs point at `data/processed/chess-pieces-2/dataset.yaml`. Prefer **`--split val`** for benchmarking (test is only 16 images).

---

## Training

Optional CUDA wheels (Linux/NVIDIA):

```bash
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

```bash
cd vision

# Apple Silicon: set training.device: "mps" in configs/roboflow/yolo/yolo.yaml and configs/roboflow/rfdetr/rfdetr.yaml
uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo/yolo.yaml
uv run python recipes/roboflow/finetune/rfdetr/main.py --config_dir configs/roboflow/rfdetr/rfdetr.yaml
```

Resume from epoch 19 → 100 (use **`last.pt`**, not `best.pt`; same Roboflow dataset):

```bash
cd vision

uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo/yolo.yaml \
  --resume ../chess_detection/runs/detect/exps/yolo_chess_pieces_2/weights/last.pt
```

Archived best weights (inference / OAK export): `data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt` (copy of run `best.pt`, best val fitness through epoch 18).

Ultralytics writes runs under `../chess_detection/runs/detect/<project>/<name>/` when training is started from `vision/` (not under `vision/exps/`). On completion, the YOLO recipe also copies to `data/checkpoints/best/best_YYYYMMDD_<run_name>.pt`.

**MPS validation appearing stuck at 52/57:** same 910-image val split; 57 batches = in-train val uses ~2× train batch (8→16). The long pause is usually **post-val plotting** and MPS memory pressure, not a frozen dataloader. `configs/roboflow/yolo/yolo.yaml` sets `plots: false` and `amp: false` for train and resume. Standalone `m.val(..., plots=False)` already finishes in ~16s on this machine.

### Validation smoke (YOLO on Roboflow split)

Quick check on the **valid** split (910 images for Chess Pieces 2) without running a full training loop.

```bash
cd vision

uv run python -c "
from ultralytics import YOLO
m = YOLO('data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt')
m.val(
    data='data/processed/chess-pieces-2/dataset.yaml',
    split='val',
    device='mps',
    plots=False,
    batch=8,
)
"
```

If MPS hangs, retry with `device='cpu'`. Expect **114** val batches at `batch=8` (910 images). In-training validation often shows **57** batches on the same split because Ultralytics uses a larger effective val batch (≈2× train `batch_size` from `configs/roboflow/yolo/yolo.yaml`).

Same metrics via the project script (writes JSON):

```bash
uv run python scripts/evaluate_models.py \
  --data data/processed/chess-pieces-2/dataset.yaml \
  --split val \
  --yolo-model ../chess_detection/runs/detect/exps/yolo_chess_pieces_2/weights/best.pt \
  --output results/yolo_chess_pieces_2_val.json
```

---

## Model comparison

Same frozen split for YOLO vs RF-DETR:

```bash
uv run python scripts/evaluate_models.py \
  --data data/processed/chess-pieces-2/dataset.yaml \
  --split val \
  --yolo-model data/checkpoints/best/best_20260110_122310.pt \
  --rfdetr-size nano \
  --output results/evaluation.json
```

---

## OAK device

Convert a trained `.pt` checkpoint to OAK `.blob`, then run detection on the camera (from `vision/`):

```bash
python scripts/convert_to_oak.py --model data/checkpoints/best/best_20260110_122310.pt

sudo python -m vision.oakd_lite.camera
sudo python -m vision.oakd_lite.chess_camera
```

Integrated pipeline (grid warp + YOLO): `scripts/chess_pipeline.py`.

---

## UDP board stream

Live OAK-D pipeline (grid warp + YOLO) publishes **one JSON object per frame** over UDP. Consumers can be a robot controller, NeuraPy bridge, or debug listener. Default port **9100** (separate from the BCI GUI on **9000** in `GUI/udp_listener.py`).

**Coordinate layers** (what we discussed):

| Frame | In stream by default? | Meaning |
|-------|------------------------|---------|
| `camera_uv` | via `board.corners_camera` | Four board corners in the 1280×720 image |
| `board_uv` | via `squares[].u/v` | Warped 800×800 top-down board pixels |
| `board_norm` | via `squares[].x/y`, `pieces[].center_norm` | 0–1 on the warped board (primary `frame` field) |
| `board_metric` | only with `--robot` | Meters on the board plane (`square_m`) |
| `tcp` | only with non-identity `T_tcp_board` in `src/robotics/hand_eye.py` | 3D points in robot TCP frame |
| `camera_3d` / full `T_tcp_cam` | not yet | Needs depth or plane lift + hand-eye |

Without `--robot`, the stream is **2D board coordinates + square names** only. **TCP pose coordinates are not included** until you set `T_tcp_board` in code.

### Commands

OpenCV UI only (no UDP):

```bash
cd vision
uv run python scripts/chess_pipeline.py
uv run python scripts/chess_pipeline.py --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt --conf 0.25
```

UDP stream (headless by default; uses latest `data/checkpoints/best/*.pt` unless `--model` is set):

```bash
cd vision

uv run python scripts/udp_streamer.py

uv run python scripts/udp_streamer.py \
  --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt \
  --conf 0.25 \
  --yolo-every 3

uv run python scripts/udp_streamer.py --udp-host 127.0.0.1 --udp-port 9100 --show

uv run python scripts/udp_streamer.py --robot
```

Two-terminal UDP debug flow:

Terminal 1 - listener:

```bash
cd vision
nc -u -l 127.0.0.1 9100
```

Terminal 2 - streamer:

```bash
cd vision
uv run python scripts/udp_streamer.py --robot
```

Val-only smoke (no camera, no UDP) after training:

```bash
cd vision
uv run python -c "
from ultralytics import YOLO
m = YOLO('data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt')
m.val(data='data/processed/chess-pieces-2/dataset.yaml', split='val', device='mps', plots=False, batch=8)
"
```

Resume interrupted YOLO training (use `last.pt`, not `best.pt`):

```bash
cd vision
uv run python recipes/roboflow/finetune/yolo/main.py --config_dir configs/roboflow/yolo/yolo.yaml \
  --resume ../chess_detection/runs/detect/exps/yolo_chess_pieces_2/weights/last.pt
```

### Notes

- **In-training val vs standalone val:** same 910-image Roboflow `valid` split; progress bar may show **57** batches during YOLO training (~2× train batch) vs **114** for `m.val(..., batch=8)`. Different batch size, not a different dataset.
- **MPS “stuck” near 52/57:** usually post-val **plotting** and GPU pressure; training config uses `plots: false` and `amp: false`. Standalone `m.val(..., plots=False)` is the isolation test.
- **Hand-eye / robot framing:** defaults live in `src/robotics/hand_eye.py` (`DEFAULT_SQUARE_M`, `DEFAULT_T_TCP_BOARD`). Use `--robot` on the UDP streamer. Override `T_tcp_board` in code when you have a measured calibration.

---

## TODO

- YOLOv11–12 training runs and OAK re-export validation
- Frame overlays (model name, piece count, latency)
- AprilTag / dynamic board calibration
- UDP schema v2: smaller payloads, `camera_3d`, `T_tcp_cam`, 6D TCP tool pose
- Optional `--udp` on `chess_pipeline.py`

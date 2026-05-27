# Chess piece datasets index (object detection)

Curated for the `vision/` pipeline: **grid warp → top-down board → YOLO / RF-DETR → square mapping → OAK export**.

Last updated: 2026-05-15 (web search + repo audit).

---

## Ranking legend

| Score | Meaning |
|-------|---------|
| **A** | Strong fit: bbox detection, ≥12 piece classes (or easy mapping), YOLO/COCO export, good for training *and* benchmark splits |
| **B** | Usable with conversion or class remapping; smaller scale or viewpoint mismatch |
| **C** | Wrong task (classification / FEN-only) or toy size; cite only, do not use as primary detector data |
| **D** | Not suitable for this pipeline |

**Pipeline fit** considers: bbox labels, class count, export format, viewpoint vs warped 800×800 board, license, and held-out test suitability.

---

## Large-scale training

| Rank | Dataset | ~Scale | Classes | Format | License | Pipeline | Benchmark |
|------|---------|--------|---------|--------|---------|----------|-----------|
| 1 | [Roboflow Universe: Chess Pieces 2](https://universe.roboflow.com/fhv/chess-pieces-2-6l8qq) | ~6.6k images (Universe listing) | 12 (`Black-bishop` … `White-rook`) | YOLOv8 zip, COCO | CC BY 4.0 | **A** | **A** — use official val/test split |
| 2 | [Roboflow Universe: 2D Chessboard and Chess Pieces](https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces) | ~5k+ images | 19 (pieces + board-related) | YOLO, COCO | CC BY 4.0 | **A** (map/filter classes) | **A** |
| 3 | [Roboflow Public: Chess Pieces (chess-full)](https://public.roboflow.com/object-detection/chess-full) | 292 images, ~2.9k boxes; aug set ~693 @ 416² | 12 standard names | YOLOv5–v12, COCO, VOC | Public Domain | **A** | **A** — fixed tripod angle; good sanity benchmark |
| 4 | [Kaggle: ninadaithal/chess-pieces-dataset](https://www.kaggle.com/datasets/ninadaithal/chess-pieces-dataset) | Small (~100–few hundred; PoC in README) | 12 (`black-camel` … `white-queen`) | YOLO in `data/raw` | Check Kaggle terms | **B** (current baseline) | **B** — too small for strong benchmark |
| 5 | [Kaggle: cookiemonsteryum/chess-piece-object-detection](https://www.kaggle.com/datasets/cookiemonsteryum/chess-piece-object-detection/data) | Medium (verify on download) | Detection, piece classes | YOLO/COCO typical | Kaggle | **B** | **B** |

---

## Supplementary or specialized

| Rank | Dataset | ~Scale | Task | Pipeline | Benchmark |
|------|---------|--------|------|----------|-----------|
| 6 | [Roboflow: Chess-Pieces-HQ](https://universe.roboflow.com/myroboflowprojects/chess-pieces-hq) | ~964 images | Often **single class** `chess_piece` | **C** for 12-class detector | **C** unless re-labeled |
| 7 | [Kaggle: anshulmehtakaggl/chess-pieces-detection-images-dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset) | Variable | **Image classification** (folder per piece) | **D** for bbox training | **D** |
| 8 | [Hugging Face: jalFaizy/detect_chess_pieces](https://huggingface.co/datasets/jalFaizy/detect_chess_pieces) | 256 images (204 train / 52 val) | 4 classes (kings/queens only) | **C** — toy / smoke tests | **C** |
| 9 | [Zenodo: Chess piece dataset (classification)](https://zenodo.org/records/6656212) | ~153 MB, multi-set | Per-square **classification**, bird's-eye | **D** for detector | **D** |
| 10 | [ChessReD](https://arxiv.org/abs/2310.04086) ([4TU data](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f)) | 10,800 real photos | **Full-board FEN**, not piece bboxes | **C** for YOLO/RF-DETR unless converted | **A** for *position* accuracy, not mAP@bbox |

---

## Research / synthetic (not primary for OAK PoC)

| Dataset | Notes | Pipeline |
|---------|-------|----------|
| Wölflein & Arandjelović (synthetic 3D, [paper](https://arxiv.org/abs/2104.14963)) | Large synthetic; piece detection / transfer learning | **B** if bbox exports obtained |
| [Kaggle: imtkaggleteam/chess-pieces-detection-image-dataset](https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset) | Verify format on download | **B** |
| [Kaggle: s4lman/chess-pieces-dataset-85x85](https://www.kaggle.com/datasets/s4lman/chess-pieces-dataset-85x85/data) | 85×85 crops | **C** — resolution mismatch |

---

## Scripted download (no browser after API key)

One-time setup:

```bash
export ROBOFLOW_API_KEY="your_key"   # from https://app.roboflow.com/settings/api
```

From the `vision/` directory:

```bash
# List presets (chess-pieces-2, chess-full-raw, chess-full-aug)
uv run python scripts/download_roboflow.py --list-presets

# Primary training set (~6.6k images)
uv run python scripts/download_roboflow.py --preset chess-pieces-2

# Optional regression sets (Public Domain, tripod / aug)
uv run python scripts/download_roboflow.py --preset chess-full-raw
uv run python scripts/download_roboflow.py --preset chess-full-aug

# Normalize class names → data/processed/<preset>/dataset.yaml
uv run python scripts/normalize_dataset.py --input data/roboflow/chess-pieces-2
```

Presets are defined in [data/roboflow_presets.yaml](../data/roboflow_presets.yaml). Override workspace/project/version if Roboflow changes slugs:

```bash
uv run python scripts/download_roboflow.py --workspace fhv --project chess-pieces-2-6l8qq --version 1 --output data/roboflow/custom
```

Kaggle remains available via `scripts/download_dataset.py` (requires `~/.kaggle/kaggle.json`).

---

## Recommended strategy for this repo

### Training (scale up)

1. `scripts/download_roboflow.py --preset chess-pieces-2` (and optionally `chess-full-aug`).
2. `scripts/normalize_dataset.py` → canonical `white-pawn` … `black-king` under `data/processed/`.
3. Map legacy Kaggle names if fine-tuning from `best_20260110_122310.pt` (`camel`→`bishop`, etc.; enabled by default in normalize).
4. Train: `configs/roboflow/yolo/yolo.yaml` and `configs/roboflow/rfdetr/rfdetr.yaml` point at `data/processed/chess-pieces-2/`.
5. `recipes/roboflow/finetune/yolo/main.py` and `recipes/roboflow/finetune/rfdetr/main.py`.

### Benchmark (YOLO vs RF-DETR)

Use a **single frozen test split** and report the same metrics for both models:

| Metric | Use |
|--------|-----|
| mAP@0.5, mAP@0.5:0.95 | Primary detector comparison (COCO eval via `pycocotools` or Ultralytics `model.val()`) |
| Per-class AP | Shows weak pieces (knights, occluded pawns) |
| Inference latency | ms/frame on warped 800×800 (CPU, MPS, CUDA, OAK) |
| Square accuracy (optional) | Map bbox center → square; compare to FEN if labels exist |

**Best benchmark sources**

| Source | Role |
|--------|------|
| Roboflow **test** split from tier-1 dataset | Standard bbox mAP benchmark |
| `vision/data/Modern_Fianchetto_Setup._Chess_game_Staunton_No._6.jpg` + manual labels | In-domain “hero” image (already used in `run_benchmark_image.py`) |
| Held-out **OAK warped frames** (`outputs/live_snapshots/`) + human labels | End-to-end system benchmark (grid + detector) |
| ChessReD test set | Only if you add a FEN/square metric; not interchangeable with bbox mAP |

`scripts/evaluate_models.py` runs YOLO `model.val()` and RF-DETR inference on the same frozen `test` split (see below).

**Do not** compare models on different test sets or mixed confidence thresholds without documenting both.

```bash
uv run python scripts/evaluate_models.py \
  --data data/processed/chess-pieces-2/dataset.yaml \
  --split test \
  --yolo-model data/checkpoints/best/best_20260110_122310.pt \
  --rfdetr-size nano \
  --output results/evaluation.json
```

---

## Class-name mapping (for merging datasets)

| Kaggle (ninadaithal) | Roboflow standard |
|--------------------|-------------------|
| black-camel / white-camel | black-bishop / white-bishop |
| black-elephant / white-elephant | black-rook / white-rook |
| black-horse / white-horse | black-knight / white-knight |
| black-pawn … black-queen | same |
| white-pawn … white-queen | same |

---

## macOS training (MPS)

### YOLO (Ultralytics)

- **Supported** in Ultralytics 8.x: set `device: "mps"` in `configs/roboflow/yolo/yolo.yaml` or CLI `device=mps`.
- Repo pin: `ultralytics>=8.0.0` (verified **8.4.50** loads YOLOv8–12 weights).
- Expect **slower than CUDA**, faster than CPU; reduce `batch_size` (e.g. 8→4) if OOM.
- If ops fail on MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1` or fall back to `device: "cpu"`.
- Docs: [Ultralytics train — Apple MPS](https://docs.ultralytics.com/modes/train/).

### RF-DETR

- Roboflow docs list `device="mps"` for `model.train()` ([training parameters](https://rfdetr.roboflow.com/learn/train/training-parameters/)).
- Use smaller variant (`nano`/`small`), lower `batch_size`, `gradient_checkpointing=True` on Mac.
- `configs/roboflow/rfdetr/rfdetr.yaml` currently only documents `cuda` / `cpu` — set `device: "mps"` when training on Mac.

### OAK export

Training on MPS is fine; **blob conversion** still runs on host (ONNX → blobconverter). Validate exported blob separately on device.

---

## YOLO versions supported (this project)

| Version | In repo today | Ultralytics 8.4.50 | Notes |
|---------|---------------|---------------------|-------|
| YOLOv8n–x | **Yes** (`configs/roboflow/yolo/yolo.yaml`, current checkpoint) | Yes | Best documented path; matches existing `.pt` / OAK scripts |
| YOLOv9 | No default | Yes (`yolov9t.pt` …) | Drop-in via `model.name` in yaml |
| YOLOv10 | No default | Yes | Drop-in |
| YOLO11 | Referenced in README / `create_demo_video.py` | Yes | Good candidate for retrain |
| YOLO12 | README TODO | Yes | Newer; verify OAK ONNX ops before committing |
| YOLO26 | — | Ultralytics docs (2026) | Newest generation; evaluate export path before OAK |

Change training model:

```yaml
# configs/roboflow/yolo/yolo.yaml
model:
  name: "yolo11n"   # or yolov8s, yolov10n, yolo12n, etc.
```

OAK pipeline historically targets **YOLOv8-style** exports; re-validate `convert_to_oak.py` after switching major versions.

---

## References

- [Roboflow chess-full overview](https://public.roboflow.com/object-detection/chess-full)
- [ChessReD paper (arXiv:2310.04086)](https://arxiv.org/abs/2310.04086)
- [Ultralytics supported models](https://docs.ultralytics.com/models/)

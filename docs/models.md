# Model families — architecture, sizes, benchmark plan

Draft for review. Numbers marked **TBD** or **ref** are placeholders until literature / upstream docs are pinned.

---

## Current repo state

| Layer | YOLO (Ultralytics) | RF-DETR |
|-------|-------------------|---------|
| Backend | `src/models/backends/ultralytics.py` → `load_yolo(path)` | `src/models/backends/rfdetr.py` → `build_rfdetr(size, checkpoint?)` |
| Training entry | `recipes/roboflow/finetune/yolo/` | `recipes/roboflow/finetune/rfdetr/` |
| Default config | `configs/roboflow/yolo/yolo.yaml` → **`yolov8n`** | `configs/roboflow/rfdetr/rfdetr.yaml` → **`base`** |
| Active checkpoint | `data/checkpoints/best/best_*_yolo_chess_pieces_2.pt` (v8n fine-tune) | not yet archived in `data/checkpoints/best/` |
| Pin | `ultralytics==8.4.56` (default ctor: `yolo26n.pt`) | `rfdetr` (Roboflow) |

**Note:** Training and deployment today use **YOLOv8n**, not YOLOv9. YOLOv9–26 are available via Ultralytics but not yet trained or benchmarked in this repo.

---

## Ultralytics interface — what “drop-in” means

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # or path to fine-tuned .pt
model.train(...) / model.val(...) / model.predict(...)
```

`load_yolo()` is a one-line wrapper around this. Any detection `.pt` or architecture `.yaml` that Ultralytics 8.x accepts works without changing the backend file. Size / version is selected via `configs/roboflow/yolo/yolo.yaml` → `model.name` or by passing a checkpoint path.

---

## YOLO generations — easy to add (detection, `n/s/m/l/x` where applicable)

| Generation | Ultralytics 8.4.56 | In repo today | Config example | OAK / ONNX |
|------------|-------------------|---------------|----------------|------------|
| YOLOv8 | yes | **default train + deploy** | `yolov8n` … `yolov8x` | validated path |
| YOLOv9 | yes | no | `yolov9t/s/m/c/e` | re-validate export |
| YOLOv10 | yes | no | `yolov10n` … `yolov10x` | re-validate export |
| YOLO11 | yes | referenced (`create_demo_video.py`) | `yolo11n` … `yolo11x` | re-validate export |
| YOLO12 | yes | README TODO | `yolo12n` … `yolo12x` | re-validate export |
| YOLO26 | yes (newest default in lib) | no | `yolo26n` … `yolo26x` | evaluate before OAK commit |

Variants **not** in scope for chess bbox benchmark unless explicitly needed: cls, seg, pose, obb, world, `yoloe-*`, RT-DETR yaml under Ultralytics (`rtdetr-*`, `yolov8-rtdetr`).

**Effort to add a new YOLO variant:** change `model.name` in yaml → train → archive `best.pt` → run eval script. No backend code change.

---

## RF-DETR sizes — implemented in repo

| Size | Backend key | Pretrained (COCO) | In repo today |
|------|-------------|-------------------|---------------|
| nano | `nano` | yes | eval CLI only |
| small | `small` | yes | — |
| base | `base` | yes | **default config** |
| medium | `medium` | yes | — |
| large | `large` | yes | — |

Checkpoint loading from fine-tuned weights is stubbed (`build_rfdetr(..., checkpoint=)` exists; `evaluate_models.py` checkpoint path still `NotImplementedError`).

---

## Architecture comparison (abstract)

| Property | YOLO (Ultralytics family) | RF-DETR |
|----------|---------------------------|---------|
| Paradigm | Single-stage, anchor-free (v8+) / evolved heads | Transformer detector (DETR lineage) |
| Backbone | CSP-style / scaled CNN (version-dependent) | ViT-style encoder (size-dependent depth) |
| Neck / fusion | PAN-FPN variants | encoder–decoder attention |
| Head | coupled or decoupled bbox + cls | object queries → boxes + classes |
| NMS | typically required at inference | end-to-end, NMS-free |
| Pretrain task | COCO detection (per variant) | COCO detection |
| Train API | `model.train(data=yaml, ...)` | `model.train(dataset_dir=..., ...)` |
| Val API | native `model.val()` → COCO metrics | upstream `.val()` **TBD** — repo uses custom loop |
| Export | ONNX, TensorRT, CoreML, OAK blob | ONNX **TBD** for OAK |
| Latency profile | lower on edge / OAK | higher; GPU-oriented |
| Size axis | `n/s/m/l/x` per generation | `nano … large` (5 steps) |

Detailed parameter counts / FLOPs: fill from upstream model cards when benchmark is fixed (**TBD**).

---

## Size grid — benchmark candidates (phase 1)

Comparable **small** models first (edge deployment target). Expand to full grid once protocol is frozen.

| ID | Family | Variant | Params (ref) | Input | Repo status |
|----|--------|---------|--------------|-------|-------------|
| Y8n | YOLOv8 | `yolov8n` | ~3M ref | 640 | trained |
| Y9t | YOLOv9 | `yolov9t` | TBD | 640 | config-only |
| Y10n | YOLOv10 | `yolov10n` | TBD | 640 | config-only |
| Y11n | YOLO11 | `yolo11n` | TBD | 640 | config-only |
| Y12n | YOLO12 | `yolo12n` | TBD | 640 | config-only |
| Y26n | YOLO26 | `yolo26n` | TBD | 640 | config-only |
| RFn | RF-DETR | `nano` | TBD | 640 | eval CLI |
| RFb | RF-DETR | `base` | TBD | 640 | default config |

Full `s/m/l/x` rows added in phase 2 after phase-1 protocol sign-off.

---

## Target benchmark table (Roboflow-style)

Reference: [Roboflow Universe — Chess Pieces](https://universe.roboflow.com/joseph-nelson/chess-pieces-new) reports **mAP@50**, **Precision**, **Recall** on a fixed test set (screenshot baseline ~98.9 / 84.2 / 99.3 — different dataset; not directly comparable to ours).

### Primary leaderboard (frozen split)

| Model | Size | mAP@50 | mAP@50:95 | Precision | Recall | conf | split | weights |
|-------|------|--------|-----------|-----------|--------|------|-------|---------|
| YOLOv8n | n | TBD | TBD | TBD | TBD | 0.25 | val | `best_*_yolo_chess_pieces_2.pt` |
| YOLO11n | n | TBD | TBD | TBD | TBD | 0.25 | val | TBD |
| YOLO26n | n | TBD | TBD | TBD | TBD | 0.25 | val | TBD |
| RF-DETR | nano | TBD | TBD | TBD | TBD | 0.25 | val | TBD |
| RF-DETR | base | TBD | TBD | TBD | TBD | 0.25 | val | TBD |

**Split choice:** prefer **`val`** (~910 images) over `test` (~16) for stable estimates (`README.md`, `docs/datasets.md`).

**External reference row (informational only):**

| Source | mAP@50 | Precision | Recall | Notes |
|--------|--------|-----------|--------|-------|
| Roboflow chess-pieces-new | 98.9% | 84.2% | 99.3% | 13 classes, different data |

### Secondary metrics (same run manifest)

| Model | eval_s | ms/img (CPU) | ms/img (CUDA) | ms/img (MPS) | ms/img (OAK) |
|-------|--------|--------------|---------------|--------------|--------------|
| … | from JSON | TBD | TBD | TBD | TBD |

### Per-class AP@50 (appendix)

| class | Y8n | Y11n | RFn | … |
|-------|-----|------|-----|---|
| white-pawn | TBD | | | |
| … | | | | |

---

## Comparability — unified benchmark (implemented)

All detectors use the same pipeline: **predict on frozen split → dual metrics** (COCO via `pycocotools` + legacy 11-point AP). See `src/evaluation/`, `configs/benchmark.yaml`, and `docs/benchmark.md`.

| Component | Location |
|-----------|----------|
| Backend protocol | `src/evaluation/backends/base.py` |
| Ultralytics / RF-DETR adapters | `src/evaluation/backends/` |
| COCO + legacy scorers | `src/evaluation/coco_eval.py` |
| Runner + manifest | `src/evaluation/runner.py`, `configs/benchmark.yaml` |
| CLI | `scripts/evaluate_models.py --manifest configs/benchmark.yaml` |

Existing entrypoint: `scripts/evaluate_models.py` → `results/benchmark/leaderboard.json`.

---

## Evaluation scenario — proposed protocol

### 1. Frozen benchmark manifest

YAML (future: `configs/benchmark.yaml`) pinning:

- `dataset_yaml`: `data/processed/chess-pieces-2/dataset.yaml`
- `split`: `val` (primary), `test` (smoke)
- `conf`: `0.25`
- `iou`: `0.5` (matching Roboflow headline metric)
- `imgsz`: `640`
- `models[]`: list of `{family, variant, weights}`

Every run writes the manifest hash into the result JSON for reproducibility.

### 2. Unified eval pipeline

```
scripts/evaluate_models.py          # keep as CLI
  └─ src/evaluation/runner.py       # future: dispatch per family
       ├─ yolo: model.val() OR export preds → common COCO eval
       └─ rfdetr: predict split → COCO JSON → pycocotools
  └─ src/metrics/coco_eval.py       # future: shared mAP@50, mAP@50:95, P, R
```

Roboflow-comparable trio: **mAP@50**, **precision**, **recall** at fixed conf (plus mAP@50:95 as secondary).

### 3. Training → eval → archive linkage

| Step | Output | Link |
|------|--------|------|
| Train | `runs/.../weights/best.pt` | `archive_best_checkpoint()` → `data/checkpoints/best/best_YYYYMMDD_<run>.pt` |
| Eval | `results/<run_id>.json` | embed `weights`, git commit, config hash, ultralytics/rfdetr version |
| Leaderboard | `results/leaderboard.csv` | append-only from eval JSON |

Run ID convention (proposal): `{family}{size}_{dataset}_{date}` e.g. `yolo11n_chess-pieces-2_20260527`.

### 4. Scheduling (batch benchmark)

Abstract options (pick one later):

- **Local matrix:** shell loop over `configs/benchmark_models.yaml` calling `evaluate_models.py`
- **Post-train hook:** recipe `main.py` calls eval on `best.pt` when `--benchmark` flag set
- **CI / nightly:** optional GitHub Action on `val` split smoke (small subset) — full matrix manual

No scheduler implemented yet.

### 5. Logging standards

| Field | Required |
|-------|----------|
| `timestamp`, `git_sha`, `ultralytics_version` / `rfdetr_version` | yes |
| `dataset_yaml`, `split`, `conf`, `iou`, `imgsz` | yes |
| `mAP50`, `mAP50_95`, `precision`, `recall` | yes |
| `per_class_ap50` | optional appendix |
| `eval_seconds`, `ms_per_image` | yes |
| `hardware`, `device` | yes |
| `notes` (metric caveats) | if any |

Structured JSON first; optional W&B / MLflow later (**TBD** — not in repo today).

### 6. Best-practice checklist

- [ ] Same split, conf, IoU for all models on leaderboard
- [ ] Fine-tuned weights only (no COCO-pretrained zero-shot for headline table)
- [ ] Document train/val/test leakage (Roboflow export splits frozen on download)
- [ ] Separate “hero image” smoke (`scripts/run_benchmark_image.py`) from quantitative leaderboard
- [ ] OAK latency tracked separately (includes blob + warp pipeline)
- [ ] External Roboflow numbers cited as reference, not as our score

---

## Context models (not in benchmark scope)

For literature / two-stage baselines only:

| Model | Backbone | Box AP (0.5:0.95) | Type |
|-------|----------|-------------------|------|
| CBNetV2 + Cascade Mask R-CNN | Dual Swin-Large | ~60.1–62.3% | Multi-stage |
| ViTDet | ViT-Huge (MAE) | ~58.7–61.3% | Two-stage |
| Cascade Mask R-CNN | Swin-Large | ~58.0% | Multi-stage |
| Co-DETR (SOTA ref) | InternImage-H | ~66.0% | Transformer |

- [Co-DETR](https://github.com/sense-x/co-detr)
- [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn)

---

## References

- [Ultralytics models](https://docs.ultralytics.com/models/)
- [RF-DETR training](https://rfdetr.roboflow.com/learn/train/training-parameters/)
- [Roboflow chess-pieces-new](https://universe.roboflow.com/joseph-nelson/chess-pieces-new)
- Repo: `docs/datasets.md` (splits), `scripts/evaluate_models.py`, `configs/roboflow/yolo/yolo.yaml`, `configs/roboflow/rfdetr/rfdetr.yaml`

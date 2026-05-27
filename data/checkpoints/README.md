# Checkpoints Directory

This directory contains model checkpoints and pretrained weights.

## Structure

- `pretrained/`: Downloaded pretrained models (ignored by git)
  - These are automatically downloaded when training starts with pretrained models
  
- `best/`: Archived best checkpoints after a completed or resumed train
  - Format: `best_YYYYMMDD_<run_name>.pt` (e.g. `best_20260515_yolo_chess_pieces_2.pt`)
  - Only checkpoints <= 50MB are auto-archived
  
- `yolo/`: Latest training outputs (ignored by git)
  - Contains `best.pt` from the most recent training run

- `oak/`: DepthAI **`.blob`** for OAK device (optional in git); **`.onnx` is gitignored**
  - Pipeline: `.pt` → ONNX (export) → `.blob` via `scripts/convert_to_oak.py`
  - You do not need to version **both** ONNX and blob. Keep **`best/*.pt`** as source of truth. Commit **`.blob`** if you deploy to OAK without re-running conversion. Omit **`.onnx`** (largest, regenerated on export).
  - Re-export: `uv run python scripts/convert_to_oak.py --model data/checkpoints/best/best_20260515_yolo_chess_pieces_2.pt`


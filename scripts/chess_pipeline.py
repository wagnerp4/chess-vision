"""
Integrated chess pipeline: grid detection → perspective warp → piece detection.

Stage 1: OAK-D camera feed with LK-tracked GridDetector produces an
         800×800 top-down warped board image each frame.
Stage 2: Ultralytics YOLO runs on the warped image to detect pieces.
         Each detection is mapped to its chess square (a1-h8).

Usage:
    cd vision
    uv run python scripts/chess_pipeline.py --model data/runs/detect/yolo_chess_pieces_2/weights/best.pt

Keys:
    q — quit
    r — reset grid tracking
    s — save snapshot of current frame + warped board

UDP stream: use scripts/udp_streamer.py
"""

import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from src.board_stream import BOARD_PX, COLS, SQUARE_PX, board_state, square_from_pixel
from src.live_pipeline import default_model, draw_detections, live_loop, run_yolo

__all__ = [
    "BOARD_PX",
    "SQUARE_PX",
    "COLS",
    "square_from_pixel",
    "run_yolo",
    "draw_detections",
    "board_state",
    "default_model",
    "live_loop",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated chess pipeline: grid warp → YOLO piece detection"
    )
    parser.add_argument("--model", default=None, help="Path to YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--yolo-every", type=int, default=3, help="Run YOLO every N frames (default 3)")
    args = parser.parse_args()

    model_path = args.model or str(default_model())
    live_loop(
        model_path=model_path,
        conf=args.conf,
        yolo_every=args.yolo_every,
        show_windows=True,
    )


if __name__ == "__main__":
    main()

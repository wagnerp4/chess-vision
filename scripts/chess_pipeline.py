import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from src.board_stream import BOARD_PX, COLS, SQUARE_PX, board_state, square_from_pixel
from src.evaluation.live_runner import default_piece_every, resolve_live_weights
from src.evaluation.types import EvalConfig
from src.inference.realtime import LiveConfig, default_model, draw_detections, live_loop, run_yolo

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
    "LiveConfig",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated chess pipeline: OAK grid warp → piece detection"
    )
    parser.add_argument("--backend", type=str, default="ultralytics", choices=["ultralytics", "rfdetr"])
    parser.add_argument("--model", "--weights", dest="weights", default=None, help="Model weights path")
    parser.add_argument("--rfdetr-size", type=str, default="large", choices=["nano", "small", "base", "medium", "large"])
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--piece-every",
        "--yolo-every",
        dest="piece_every",
        type=int,
        default=None,
        help="Run piece detector every N frames (default: 3 ultralytics, 6 rfdetr)",
    )
    parser.add_argument("--grid-every", type=int, default=2, help="Grid detect cadence when tracking")
    parser.add_argument("--redetect-every", type=int, default=60, help="Force grid re-anchor interval")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = manual)")
    args = parser.parse_args()

    piece_every = args.piece_every or default_piece_every(args.backend)
    weights = resolve_live_weights(args.backend, args.weights, root=VISION_ROOT)

    config = LiveConfig(
        backend=args.backend,
        weights=weights,
        model_label=Path(weights).name,
        rfdetr_size=args.rfdetr_size if args.backend == "rfdetr" else None,
        eval=EvalConfig(conf=args.conf),
        piece_every=piece_every,
        grid_every=args.grid_every,
        redetect_every=args.redetect_every,
        show_windows=True,
        duration_sec=args.duration,
    )
    live_loop(config, root=VISION_ROOT)


if __name__ == "__main__":
    main()

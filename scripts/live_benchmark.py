import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.evaluation.live_runner import (
    default_piece_every,
    resolve_live_weights,
    run_live_benchmark,
    write_live_benchmark,
)
from src.evaluation.types import EvalConfig
from src.inference.realtime import LiveConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live OAK-D Lite benchmark: grid warp + backend-agnostic piece detection"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "rfdetr"],
        help="Detector backend (default: ultralytics)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model weights path (.pt or RF-DETR .ckpt/.pth)",
    )
    parser.add_argument(
        "--rfdetr-size",
        type=str,
        default="large",
        choices=["nano", "small", "base", "medium", "large"],
        help="RF-DETR variant when --backend rfdetr",
    )
    parser.add_argument("--label", type=str, default=None, help="Human-readable model label")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou-nms", type=float, default=0.45, help="NMS IoU (ultralytics)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--piece-every",
        type=int,
        default=None,
        help="Run piece detector every N camera frames (bbox update rate)",
    )
    parser.add_argument(
        "--grid-every",
        type=int,
        default=2,
        help="Attempt full grid detect every N frames when LK tracking is active",
    )
    parser.add_argument(
        "--redetect-every",
        type=int,
        default=60,
        help="Force grid re-anchor every N frames",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then stop (0 = until q)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="OAK RGB frame width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="OAK RGB frame height",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/live_benchmark.json",
        help="Latency JSON output path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV windows (default: headless benchmark)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing JSON report",
    )
    args = parser.parse_args()

    piece_every = args.piece_every
    if piece_every is None:
        piece_every = default_piece_every(args.backend)

    weights = resolve_live_weights(args.backend, args.weights, root=VISION_ROOT)
    label = args.label or f"{args.backend} {Path(weights).name}"

    config = LiveConfig(
        backend=args.backend,
        weights=weights,
        model_label=label,
        rfdetr_size=args.rfdetr_size if args.backend == "rfdetr" else None,
        eval=EvalConfig(
            conf=args.conf,
            iou_nms=args.iou_nms,
            imgsz=args.imgsz,
        ),
        piece_every=piece_every,
        grid_every=args.grid_every,
        redetect_every=args.redetect_every,
        camera_size=(args.camera_width, args.camera_height),
        show_windows=args.show,
        duration_sec=args.duration,
    )

    print(
        f"Live benchmark: backend={args.backend}  piece_every={piece_every}  "
        f"duration={args.duration or 'until q'}s"
    )
    payload = run_live_benchmark(config, root=VISION_ROOT)

    latency = payload["latency"]
    print(
        f"Frames={latency['frames_total']}  board={latency['frames_with_board']}  "
        f"infer_runs={latency['inference_runs']}"
    )
    if latency.get("fps_mean"):
        print(f"Camera FPS mean={latency['fps_mean']:.1f}")
    infer = latency.get("inference_ms", {})
    if infer.get("mean") is not None:
        print(
            f"Inference ms mean={infer['mean']:.1f}  p50={infer.get('p50', 0):.1f}  "
            f"p95={infer.get('p95', 0):.1f}"
        )

    if not args.no_write:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = VISION_ROOT / out_path
        write_live_benchmark(payload, out_path)
        print(f"Report saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

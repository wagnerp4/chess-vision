"""
Live chess board stream over UDP (JSON datagrams).

Runs the OAK-D grid + YOLO pipeline and publishes board/square/piece coordinates.
Optional robot block (board_metric / tcp) when --hand-eye config is provided.

Usage:
    cd vision
    uv run python scripts/udp_streamer.py
    uv run python scripts/udp_streamer.py --udp-port 9100 --show
    uv run python scripts/udp_streamer.py --hand-eye configs/hand_eye.example.yaml

Listen (another terminal):
    nc -u -l 127.0.0.1 9100
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Any

VISION_ROOT = Path(__file__).resolve().parent.parent
if str(VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISION_ROOT))

from src.board_stream import UdpPublisher, build_payload, load_hand_eye
from src.live_pipeline import default_model, live_loop

SCHEMA_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess board UDP stream (schema v1)")
    parser.add_argument("--model", default=None, help="YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    parser.add_argument("--yolo-every", type=int, default=3, help="Run YOLO every N frames")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP destination host")
    parser.add_argument("--udp-port", type=int, default=9100, help="UDP destination port")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV windows (default: headless stream only)",
    )
    parser.add_argument(
        "--hand-eye",
        default=None,
        help="YAML with square_m and optional T_tcp_board (4x4) for robot coordinates",
    )
    args = parser.parse_args()

    hand_eye = load_hand_eye(args.hand_eye)
    publisher = UdpPublisher(host=args.udp_host, port=args.udp_port)
    model_path = args.model or str(default_model())
    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        print(f"\n{name} received. Stopping UDP stream...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"UDP stream → {args.udp_host}:{args.udp_port}  schema={SCHEMA_VERSION}")
    if hand_eye:
        print(f"Robot framing enabled (square_m={hand_eye['square_m']})")
    else:
        print("Robot block omitted (pass --hand-eye for board_metric / tcp)")
    if not args.show:
        print("Headless mode. Press Ctrl+C to stop.")

    def on_frame(meta: dict[str, Any], _frame, _warped) -> None:
        payload = build_payload(
            seq=0,
            frame_id=int(meta["frame_count"]),
            valid=bool(meta["valid"]),
            corners_camera=meta["corners"],
            homography=meta["homography"],
            detections=meta["detections"],
            pieces_stale=bool(meta["pieces_stale"]),
            hand_eye=hand_eye,
        )
        ok = publisher.send(payload)
        if meta["frame_count"] % 30 == 0:
            status = "sent" if ok else "dropped"
            valid = meta["valid"]
            n = len(meta["detections"])
            print(f"frame={meta['frame_count']} valid={valid} pieces={n} udp={status}")

    try:
        live_loop(
            model_path=model_path,
            conf=args.conf,
            yolo_every=args.yolo_every,
            show_windows=args.show,
            on_frame=on_frame,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        publisher.close()
        print("Stopped.")


if __name__ == "__main__":
    main()

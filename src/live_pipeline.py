from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

if sys.platform != "win32":
    import pathlib

    pathlib.WindowsPath = Path
    sys.modules["pathlib"].WindowsPath = Path

import cv2
import depthai as dai
import numpy as np

from src.board_stream import BOARD_PX, board_state, square_from_pixel
from src.grid_detector import GridDetector

DETECT_SCALE = 0.5
DETECT_EVERY = 2
REDETECT_EVERY = 60

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
MAX_LK_ERROR = 15.0
MAX_LK_JUMP = 150.0


def run_yolo(model, warped: np.ndarray, conf: float) -> list[dict]:
    results = model.predict(warped, conf=conf, verbose=False)
    detections = []
    for det in results[0]:
        boxes = det.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "class": model.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "bbox": bbox,
                "square": square_from_pixel(cx, cy),
            })
    return detections


def draw_detections(warped: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = warped.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f"{d['class']} {d['confidence']:.2f} [{d['square']}]"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1, lh + 6)
        cv2.rectangle(out, (x1, ly - lh - 6), (x1 + lw, ly), (0, 255, 0), -1)
        cv2.putText(out, label, (x1, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return out


def default_model() -> Path:
    vision_root = Path(__file__).resolve().parent.parent
    candidates = [
        vision_root / "data" / "runs" / "detect" / "yolo_chess_pieces_2" / "weights" / "best.pt",
        vision_root / "data" / "checkpoints" / "yolo" / "best.pt",
        vision_root / "data" / "checkpoints" / "best" / "best_20260515_yolo_chess_pieces_2.pt",
        vision_root / "data" / "checkpoints" / "best" / "best_20260110_122310.pt",
        vision_root / "data" / "checkpoints" / "best",
    ]
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir():
            pts = sorted(c.glob("*.pt"))
            if pts:
                return pts[-1]
    raise FileNotFoundError("No .pt model found. Provide --model /path/to/weights.pt")


FrameCallback = Callable[[dict[str, Any], np.ndarray | None, np.ndarray | None], None]


def live_loop(
    model_path: str,
    conf: float = 0.25,
    yolo_every: int = 3,
    show_windows: bool = True,
    on_frame: FrameCallback | None = None,
    snap_dir: Path | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    from ultralytics import YOLO

    if stop_event is None:
        stop_event = threading.Event()

    detector = GridDetector(output_size=BOARD_PX)
    yolo = YOLO(model_path)
    print(f"YOLO model loaded: {model_path}")
    print(f"Classes: {list(yolo.names.values())}")

    prev_gray = None
    tracked_pts = None
    smoothed = None
    frames_since_redetect = 0
    last_detections: list[dict] = []
    last_yolo_frame = 0

    if snap_dir is None:
        snap_dir = Path(__file__).parent.parent / "outputs" / "live_snapshots"

    device = dai.Device()
    with dai.Pipeline(device) as pipeline:
        sockets = device.getConnectedCameras()
        cam = pipeline.create(dai.node.Camera).build(sockets[0])
        rgb_queue = cam.requestOutput(
            size=(1280, 720),
            type=dai.ImgFrame.Type.BGR888p,
        ).createOutputQueue()

        pipeline.start()
        print("OAK-D Lite connected.")

        frame_count = 0

        while pipeline.isRunning() and not stop_event.is_set():
            video_in = rgb_queue.tryGet()
            if video_in is None:
                time.sleep(0.01)
                continue
            frame = video_in.getCvFrame()
            frame_count += 1

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            frames_since_redetect += 1
            lk_ok = False

            if prev_gray is not None and tracked_pts is not None:
                new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, tracked_pts, None, **LK_PARAMS
                )
                tracked = status.flatten() == 1
                if tracked.all() and np.max(error[tracked]) < MAX_LK_ERROR:
                    drift = np.max(np.linalg.norm(new_pts.reshape(4, 2) - smoothed, axis=1))
                    if drift < MAX_LK_JUMP:
                        smoothed = new_pts.reshape(4, 2).astype(np.float32)
                        tracked_pts = new_pts
                        lk_ok = True

            if not lk_ok or frames_since_redetect >= REDETECT_EVERY:
                if frame_count % DETECT_EVERY == 0:
                    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                    result = detector.process_image_debug(small)
                    if result["success"]:
                        smoothed = (result["board_corners"] / DETECT_SCALE).astype(np.float32)
                        tracked_pts = smoothed.reshape(4, 1, 2)
                        frames_since_redetect = 0
                    elif not lk_ok:
                        smoothed = None
                        tracked_pts = None

            prev_gray = curr_gray.copy()

            warped = None
            homography = None
            annotated = None

            if smoothed is not None:
                display = detector.draw_grid_from_corners(frame, smoothed)
                warped, homography = detector.warp_board(frame, smoothed)

                if frame_count % yolo_every == 0:
                    last_detections = run_yolo(yolo, warped, conf)
                    last_yolo_frame = frame_count

                pieces_stale = (frame_count - last_yolo_frame) > 0 and frame_count % yolo_every != 0

                if show_windows:
                    annotated = detector.draw_grid_on_warped(warped)
                    annotated = draw_detections(annotated, last_detections)
                    cv2.imshow("Warped board — piece detection", annotated)

                if last_detections:
                    overlay = f"Pieces: {len(last_detections)}"
                    cv2.putText(
                        display, overlay, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
                    )

                if on_frame is not None:
                    on_frame(
                        {
                            "frame_count": frame_count,
                            "valid": True,
                            "corners": smoothed.copy(),
                            "homography": homography.copy(),
                            "detections": list(last_detections),
                            "pieces_stale": pieces_stale,
                        },
                        frame,
                        warped,
                    )
            else:
                if show_windows:
                    cv2.putText(
                        display, "No board detected",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2,
                    )
                if on_frame is not None:
                    on_frame(
                        {
                            "frame_count": frame_count,
                            "valid": False,
                            "corners": None,
                            "homography": None,
                            "detections": [],
                            "pieces_stale": False,
                        },
                        frame,
                        None,
                    )

            if show_windows:
                cv2.imshow("OAK-D Lite — Chess Pipeline", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord("q"):
                print("Quitting.")
                break

            if key == ord("r"):
                smoothed, tracked_pts, prev_gray = None, None, None
                frames_since_redetect = 0
                last_detections = []
                print("Reset.")

            if key == ord("s") and smoothed is not None and show_windows and annotated is not None:
                snap_dir.mkdir(parents=True, exist_ok=True)
                tag = f"{frame_count:06d}"
                cv2.imwrite(str(snap_dir / f"{tag}_grid.jpg"), display)
                cv2.imwrite(str(snap_dir / f"{tag}_warped.jpg"), annotated)
                state_str = "\n".join(
                    f"{sq}: {pc}" for sq, pc in sorted(board_state(last_detections).items())
                )
                (snap_dir / f"{tag}_state.txt").write_text(state_str)
                print(f"Snapshot saved → {snap_dir}/{tag}_*")

    if show_windows:
        cv2.destroyAllWindows()

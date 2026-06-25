from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
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
from src.evaluation.backends.registry import get_backend
from src.evaluation.types import Detection, EvalConfig, ModelSpec
from src.grid_detector import GridDetector

DETECT_SCALE = 0.5
DEFAULT_GRID_EVERY = 2
DEFAULT_REDETECT_EVERY = 60
DEFAULT_PIECE_EVERY = {
    "ultralytics": 3,
    "rfdetr": 6,
}

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
MAX_LK_ERROR = 15.0
MAX_LK_JUMP = 150.0

FrameCallback = Callable[[dict[str, Any], np.ndarray | None, np.ndarray | None], None]


@dataclass
class LiveConfig:
    backend: str = "ultralytics"
    weights: str = ""
    model_label: str = ""
    rfdetr_size: str | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    piece_every: int = 3
    grid_every: int = DEFAULT_GRID_EVERY
    redetect_every: int = DEFAULT_REDETECT_EVERY
    camera_size: tuple[int, int] = (1280, 720)
    show_windows: bool = True
    duration_sec: float = 0.0
    snap_dir: Path | None = None


@dataclass
class LiveLatencyTracker:
    loop_ms: list[float] = field(default_factory=list)
    frame_interval_ms: list[float] = field(default_factory=list)
    grid_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    frames_total: int = 0
    frames_with_board: int = 0
    inference_runs: int = 0
    _last_frame_ts: float | None = None

    def on_frame_start(self) -> float:
        now = time.perf_counter()
        if self._last_frame_ts is not None:
            self.frame_interval_ms.append((now - self._last_frame_ts) * 1000.0)
        self._last_frame_ts = now
        return now

    def on_frame_end(self, t0: float) -> None:
        self.loop_ms.append((time.perf_counter() - t0) * 1000.0)
        self.frames_total += 1

    def on_grid(self, elapsed_ms: float) -> None:
        self.grid_ms.append(elapsed_ms)

    def on_inference(self, elapsed_ms: float) -> None:
        self.inference_ms.append(elapsed_ms)
        self.inference_runs += 1

    def summary(self, piece_every: int) -> dict[str, Any]:
        def pct(values: list[float], q: float) -> float | None:
            if not values:
                return None
            return float(np.percentile(values, q))

        def mean(values: list[float]) -> float | None:
            if not values:
                return None
            return float(np.mean(values))

        fps = None
        if self.frame_interval_ms:
            fps = 1000.0 / mean(self.frame_interval_ms)

        infer_fps = None
        if self.inference_ms and self.frame_interval_ms:
            infer_fps = self.inference_runs / max(
                sum(self.frame_interval_ms) / 1000.0, 1e-6
            )

        return {
            "frames_total": self.frames_total,
            "frames_with_board": self.frames_with_board,
            "inference_runs": self.inference_runs,
            "piece_every": piece_every,
            "fps_mean": fps,
            "inference_fps_mean": infer_fps,
            "loop_ms": {
                "mean": mean(self.loop_ms),
                "p50": pct(self.loop_ms, 50),
                "p95": pct(self.loop_ms, 95),
            },
            "frame_interval_ms": {
                "mean": mean(self.frame_interval_ms),
                "p50": pct(self.frame_interval_ms, 50),
                "p95": pct(self.frame_interval_ms, 95),
            },
            "grid_ms": {
                "mean": mean(self.grid_ms),
                "p50": pct(self.grid_ms, 50),
                "p95": pct(self.grid_ms, 95),
                "count": len(self.grid_ms),
            },
            "inference_ms": {
                "mean": mean(self.inference_ms),
                "p50": pct(self.inference_ms, 50),
                "p95": pct(self.inference_ms, 95),
                "count": len(self.inference_ms),
            },
        }


def default_model() -> Path:
    vision_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        vision_root / "data" / "checkpoints" / "best" / "best_20260527_yolo26n_chess_pieces_2.pt",
        vision_root / "data" / "checkpoints" / "best" / "best_20260515_yolo_chess_pieces_2.pt",
        vision_root / "data" / "runs" / "detect" / "yolo_chess_pieces_2" / "weights" / "best.pt",
        vision_root / "data" / "checkpoints" / "yolo" / "best.pt",
        vision_root / "data" / "checkpoints" / "best",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            pts = sorted(candidate.glob("*.pt"))
            if pts:
                return pts[-1]
    raise FileNotFoundError("No .pt model found. Provide --weights /path/to/weights")


def detections_to_board_dict(detections: list[Detection]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        out.append({
            "class": det.class_name,
            "confidence": det.confidence,
            "bbox": det.bbox,
            "square": square_from_pixel(cx, cy),
        })
    return out


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


def run_yolo(model, warped: np.ndarray, conf: float) -> list[dict]:
    results = model.predict(warped, conf=conf, verbose=False)
    detections: list[dict] = []
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


def build_live_backend(config: LiveConfig, root: Path | None = None):
    backend = get_backend(
        ModelSpec(
            id="live",
            backend=config.backend,
            weights=config.weights,
            label=config.model_label or config.backend,
            size=config.rfdetr_size,
        )
    )
    backend.load(
        ModelSpec(
            id="live",
            backend=config.backend,
            weights=config.weights,
            label=config.model_label or config.backend,
            size=config.rfdetr_size,
        ),
        root=root,
    )
    return backend


def live_loop(
    config: LiveConfig,
    on_frame: FrameCallback | None = None,
    stop_event: threading.Event | None = None,
    latency: LiveLatencyTracker | None = None,
    root: Path | None = None,
) -> LiveLatencyTracker:
    if stop_event is None:
        stop_event = threading.Event()
    if latency is None:
        latency = LiveLatencyTracker()

    detector = GridDetector(output_size=BOARD_PX)
    piece_backend = build_live_backend(config, root=root)
    print(f"Live backend: {config.backend}  weights={config.weights}")
    print(
        f"piece_every={config.piece_every}  grid_every={config.grid_every}  "
        f"redetect_every={config.redetect_every}  conf={config.eval.conf}"
    )

    prev_gray = None
    tracked_pts = None
    smoothed = None
    frames_since_redetect = 0
    last_detections: list[dict] = []
    last_piece_frame = 0

    snap_dir = config.snap_dir
    if snap_dir is None:
        snap_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "live_snapshots"

    started = time.perf_counter()
    device = dai.Device()
    cam_w, cam_h = config.camera_size

    with dai.Pipeline(device) as pipeline:
        sockets = device.getConnectedCameras()
        cam = pipeline.create(dai.node.Camera).build(sockets[0])
        rgb_queue = cam.requestOutput(
            size=(cam_w, cam_h),
            type=dai.ImgFrame.Type.BGR888p,
        ).createOutputQueue()

        pipeline.start()
        print("OAK-D Lite connected.")

        frame_count = 0

        while pipeline.isRunning() and not stop_event.is_set():
            if config.duration_sec > 0 and (time.perf_counter() - started) >= config.duration_sec:
                print(f"Duration limit reached ({config.duration_sec}s).")
                break

            t_loop = latency.on_frame_start()
            video_in = rgb_queue.tryGet()
            if video_in is None:
                time.sleep(0.01)
                latency.on_frame_end(t_loop)
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

            if not lk_ok or frames_since_redetect >= config.redetect_every:
                if frame_count % config.grid_every == 0:
                    t_grid = time.perf_counter()
                    small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                    result = detector.process_image_debug(small)
                    latency.on_grid((time.perf_counter() - t_grid) * 1000.0)
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
                latency.frames_with_board += 1
                display = detector.draw_grid_from_corners(frame, smoothed)
                warped, homography = detector.warp_board(frame, smoothed)

                if frame_count % config.piece_every == 0:
                    t_infer = time.perf_counter()
                    raw_dets = piece_backend.predict_bgr(warped, config.eval)
                    latency.on_inference((time.perf_counter() - t_infer) * 1000.0)
                    last_detections = detections_to_board_dict(raw_dets)
                    last_piece_frame = frame_count

                pieces_stale = (
                    (frame_count - last_piece_frame) > 0
                    and frame_count % config.piece_every != 0
                )

                if config.show_windows:
                    annotated = detector.draw_grid_on_warped(warped)
                    annotated = draw_detections(annotated, last_detections)
                    cv2.imshow("Warped board — piece detection", annotated)

                if last_detections:
                    overlay = f"Pieces: {len(last_detections)}"
                    cv2.putText(
                        display, overlay, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
                    )
                infer_age = frame_count - last_piece_frame
                status_line = (
                    f"piece_every={config.piece_every}  stale={pieces_stale}  "
                    f"infer_age={infer_age}f"
                )
                cv2.putText(
                    display, status_line, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1,
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
                            "piece_every": config.piece_every,
                        },
                        frame,
                        warped,
                    )
            else:
                if config.show_windows:
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
                            "piece_every": config.piece_every,
                        },
                        frame,
                        None,
                    )

            if config.show_windows:
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

            if (
                key == ord("s")
                and smoothed is not None
                and config.show_windows
                and annotated is not None
            ):
                snap_dir.mkdir(parents=True, exist_ok=True)
                tag = f"{frame_count:06d}"
                cv2.imwrite(str(snap_dir / f"{tag}_grid.jpg"), display)
                cv2.imwrite(str(snap_dir / f"{tag}_warped.jpg"), annotated)
                state_str = "\n".join(
                    f"{sq}: {pc}" for sq, pc in sorted(board_state(last_detections).items())
                )
                (snap_dir / f"{tag}_state.txt").write_text(state_str)
                print(f"Snapshot saved → {snap_dir}/{tag}_*")

            latency.on_frame_end(t_loop)

    if config.show_windows:
        cv2.destroyAllWindows()

    return latency

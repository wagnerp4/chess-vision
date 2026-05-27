from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

CHESS_CLASSES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]
CHESS_CLASSES_12 = [
    "black-bishop", "black-rook", "black-knight", "black-king", "black-pawn", "black-queen",
    "white-bishop", "white-rook", "white-knight", "white-king", "white-pawn", "white-queen",
]
COLORS_12 = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 128), 
    (128, 0, 255), (255, 255, 128), (128, 255, 255), (255, 128, 255)
]


def _vision_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_blob_path(blob_path: str | Path | None) -> Path:
    if blob_path is not None:
        p = Path(blob_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Blob not found: {p}")

    root = _vision_root()
    oak_dir = root / "data" / "checkpoints" / "oak"
    if oak_dir.exists():
        blobs = sorted(oak_dir.glob("*.blob"), key=lambda p: p.stat().st_mtime, reverse=True)
        if blobs:
            return blobs[-1]

    for base in [Path.cwd(), Path.cwd() / "vision", root]:
        cand = base / "data" / "checkpoints" / "oak"
        if cand.exists():
            blobs = sorted(cand.glob("*.blob"), key=lambda p: p.stat().st_mtime, reverse=True)
            if blobs:
                return blobs[-1]

    raise FileNotFoundError(
        "No .blob found in data/checkpoints/oak. "
        "Convert a checkpoint: python scripts/convert_to_oak.py --model data/checkpoints/best/best_*.pt"
    )


def create_pipeline(
    pipeline: dai.Pipeline,
    blob_path: str | Path,
    input_size: int = 640,
    camera_socket: dai.CameraBoardSocket | None = None,
) -> tuple:
    socket = camera_socket if camera_socket is not None else dai.CameraBoardSocket.CAM_A
    cam = pipeline.create(dai.node.Camera).build(socket)
    cam_out = cam.requestOutput((input_size, input_size), type=dai.ImgFrame.Type.BGR888p)
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    cam_out.link(nn.input)
    return cam, nn


def _get_tensor_from_nn_data(nn_data) -> np.ndarray:
    try:
        t = nn_data.getFirstTensor(dequantize=True)
        return np.asarray(t, dtype=np.float32)
    except Exception:
        pass
    names = nn_data.getAllLayerNames()
    if names:
        t = nn_data.getTensor(names[0], dequantize=True)
        return np.asarray(t, dtype=np.float32)
    raise ValueError("NNData has no readable tensor")


def decode_yolo_output(
    nn_data,
    frame_shape: tuple,
    input_size: int = 640,
    conf_thresh: float = 0.25,
    class_names: list[str] | None = None,
) -> list[dict]:
    raw = _get_tensor_from_nn_data(nn_data)
    if raw.size == 0:
        return []
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        raw = raw.T
    if raw.ndim != 2 or raw.shape[1] < 5:
        return []

    C = raw.shape[1]
    if C == 16:
        num_classes = 12
        obj = 1.0
        scores = raw[:, 4:16]
        names = class_names or CHESS_CLASSES_12
    else:
        num_classes = len(CHESS_CLASSES)
        min_dims = 4 + num_classes
        if C < min_dims:
            return []
        has_objectness = C >= 4 + 1 + num_classes
        if has_objectness and C > 5 + num_classes:
            obj = raw[:, 4]
            scores = raw[:, 5 : 5 + num_classes]
        else:
            obj = np.ones(raw.shape[0], dtype=raw.dtype)
            scores = raw[:, 4 : 4 + num_classes]
        names = class_names or CHESS_CLASSES

    boxes = []
    for i in range(raw.shape[0]):
        row = raw[i]
        xc, yc, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if C == 16:
            sc = scores[i]
            conf = float(np.max(sc))
            cid = int(np.argmax(sc))
        else:
            obj_val = float(obj[i]) if hasattr(obj, "__len__") and i < len(obj) else 1.0
            sc = np.asarray(scores[i], dtype=np.float64) * obj_val
            conf = float(np.max(sc))
            cid = int(np.argmax(sc))
        if conf < conf_thresh:
            continue

        h_f, w_f = frame_shape[0], frame_shape[1]
        sx = w_f / input_size
        sy = h_f / input_size
        x1 = int((xc - w / 2) * sx)
        y1 = int((yc - h / 2) * sy)
        x2 = int((xc + w / 2) * sx)
        y2 = int((yc + h / 2) * sy)
        x1 = max(0, min(x1, w_f))
        y1 = max(0, min(y1, h_f))
        x2 = max(0, min(x2, w_f))
        y2 = max(0, min(y2, h_f))

        if x2 > x1 and y2 > y1:
            cid_clamped = min(cid, len(names) - 1)
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cid,
                "class_name": names[cid_clamped],
            })

    return boxes


def draw_detections(frame: np.ndarray, detections: list[dict], colors: list[tuple] | None = None) -> np.ndarray:
    pal = colors if colors is not None else (COLORS_12 if (detections and max(d["class_id"] for d in detections) >= 6) else COLORS)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cid = d["class_id"]
        color = pal[cid % len(pal)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ly = max(y1, lh + 10)
        cv2.rectangle(frame, (x1, ly - lh - 10), (x1 + lw, ly), color, -1)
        cv2.putText(frame, label, (x1, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def run(
    blob_path: str | Path | None = None,
    input_size: int = 640,
    conf_threshold: float = 0.25,
) -> None:
    blob = find_blob_path(blob_path)

    try:
        device = dai.Device()
        with dai.Pipeline(device) as pipeline:
            sockets = device.getConnectedCameras()
            camera_socket = sockets[0] if sockets else dai.CameraBoardSocket.CAM_A
            cam, nn = create_pipeline(pipeline, blob, input_size, camera_socket=camera_socket)
            nn_q = nn.out.createOutputQueue(maxSize=4, blocking=False)
            pt_q = None
            if hasattr(nn, "passthrough"):
                pt_q = nn.passthrough.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            time.sleep(2.0)

            fallback = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            fallback[:] = (96, 96, 96)
            last_dets = []
            last_pt_frame = None

            log = logging.getLogger("chess_camera")
            log.setLevel(logging.INFO)
            if not log.handlers:
                h = logging.StreamHandler(sys.stderr)
                h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
                log.addHandler(h)
            log.info("blob=%s camera_socket=%s frames=%s", blob, camera_socket, "nn.passthrough" if pt_q else "fallback")
            log.info("run without sudo if the window stays black (sudo can break GUI on macOS)")

            it = 0
            nn_count = 0

            while pipeline.isRunning():
                pt_frame = pt_q.tryGet() if pt_q is not None else None
                nn_data = nn_q.tryGet()
                it += 1

                if nn_data is not None:
                    nn_count += 1
                    dets = decode_yolo_output(nn_data, (input_size, input_size), input_size, conf_threshold)
                    last_dets = [d for d in dets if d["confidence"] >= conf_threshold]
                    if nn_count <= 3 or nn_count % 100 == 0:
                        try:
                            t = _get_tensor_from_nn_data(nn_data)
                            log.info("nn_data #%d shape=%s ndet=%d", nn_count, t.shape, len(last_dets))
                        except Exception as e:
                            log.info("nn_data #%d ndet=%d (tensor read: %s)", nn_count, len(last_dets), e)

                if it % 120 == 0:
                    log.info("iters=%d nn_received=%d last_dets=%d", it, nn_count, len(last_dets))

                if pt_frame is not None:
                    last_pt_frame = pt_frame
                src = pt_frame or last_pt_frame
                base = src.getCvFrame().copy() if src is not None else fallback.copy()
                display = draw_detections(base, last_dets)
                cv2.imshow("chess_camera", display)

                delay = 33 if src is not None else 1
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
    except RuntimeError as e:
        if "X_LINK" in str(e):
            print(f"Device error: {e}")
            print("Check USB (3.0), cable, and depthai install.")
        else:
            raise
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess piece detection on OAK using YOLO blob")
    parser.add_argument("--model", default=None, help="Path to .blob (default: data/checkpoints/oak/*.blob)")
    parser.add_argument("--input-size", type=int, default=640, help="Model input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run(blob_path=args.model, input_size=args.input_size, conf_threshold=args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

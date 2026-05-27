from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8
COLS = "abcdefgh"
ROWS = "87654321"


def square_name(col: int, row: int) -> str:
    return f"{COLS[col]}{ROWS[row]}"


def board_state(detections: list[dict[str, Any]]) -> dict[str, str]:
    return {d["square"]: d["class"] for d in detections}


def square_from_pixel(x: float, y: float) -> str:
    col = min(int(x) // SQUARE_PX, 7)
    row = min(int(y) // SQUARE_PX, 7)
    return square_name(col, row)


def all_square_centers() -> dict[str, dict[str, float]]:
    squares: dict[str, dict[str, float]] = {}
    for row in range(8):
        for col in range(8):
            name = square_name(col, row)
            u = (col + 0.5) * SQUARE_PX
            v = (row + 0.5) * SQUARE_PX
            squares[name] = {
                "u": round(u, 2),
                "v": round(v, 2),
                "x": round(u / BOARD_PX, 6),
                "y": round(v / BOARD_PX, 6),
            }
    return squares


def board_metric_point(norm_x: float, norm_y: float, square_m: float) -> list[float]:
    return [
        float(norm_x * 8.0 * square_m),
        float(norm_y * 8.0 * square_m),
        0.0,
    ]


def load_hand_eye(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    square_m = float(data.get("square_m", 0.037))
    out: dict[str, Any] = {"square_m": square_m}
    if "T_tcp_board" in data:
        out["T_tcp_board"] = np.asarray(data["T_tcp_board"], dtype=float).reshape(4, 4)
    if "T_tcp_cam" in data:
        out["T_tcp_cam"] = np.asarray(data["T_tcp_cam"], dtype=float).reshape(4, 4)
    return out


def apply_homogeneous(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_h = np.asarray([p[0], p[1], p[2], 1.0], dtype=float)
    out = T @ p_h
    return out[:3] / out[3] if abs(out[3]) > 1e-12 else out[:3]


def enrich_robot_block(
    squares: dict[str, dict[str, float]],
    pieces: list[dict[str, Any]],
    hand_eye: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if hand_eye is None:
        return None
    square_m = hand_eye["square_m"]
    T_tcp_board = hand_eye.get("T_tcp_board")
    squares_tcp: dict[str, list[float]] = {}
    squares_board_m: dict[str, list[float]] = {}
    for name, sq in squares.items():
        p_board = board_metric_point(sq["x"], sq["y"], square_m)
        squares_board_m[name] = [round(v, 6) for v in p_board]
        if T_tcp_board is not None:
            p_tcp = apply_homogeneous(T_tcp_board, np.asarray(p_board, dtype=float))
            squares_tcp[name] = [round(float(v), 6) for v in p_tcp]
    pieces_tcp = []
    for piece in pieces:
        sq = squares.get(piece["square"], {})
        if not sq:
            continue
        p_board = board_metric_point(sq["x"], sq["y"], square_m)
        entry: dict[str, Any] = {
            "square": piece["square"],
            "class": piece["class"],
            "center_board_m": [round(v, 6) for v in p_board],
        }
        if T_tcp_board is not None:
            p_tcp = apply_homogeneous(T_tcp_board, np.asarray(p_board, dtype=float))
            entry["center_tcp"] = [round(float(v), 6) for v in p_tcp]
        pieces_tcp.append(entry)
    block: dict[str, Any] = {
        "enabled": True,
        "frame": "tcp" if T_tcp_board is not None else "board_metric",
        "square_m": square_m,
        "squares_board_m": squares_board_m,
        "pieces_board_m": pieces_tcp,
    }
    if T_tcp_board is not None:
        block["squares_tcp"] = squares_tcp
    if hand_eye.get("T_tcp_cam") is not None:
        block["T_tcp_cam"] = hand_eye["T_tcp_cam"].tolist()
    return block


def piece_records(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pieces = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pieces.append({
            "id": i,
            "class": d["class"],
            "square": d["square"],
            "confidence": round(float(d["confidence"]), 4),
            "center_board": [round(cx, 2), round(cy, 2)],
            "center_norm": [round(cx / BOARD_PX, 6), round(cy / BOARD_PX, 6)],
            "bbox_board": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    return pieces


def build_payload(
    *,
    seq: int,
    frame_id: int,
    valid: bool,
    corners_camera: np.ndarray | None,
    homography: np.ndarray | None,
    detections: list[dict[str, Any]],
    pieces_stale: bool,
    hand_eye: dict[str, Any] | None = None,
    primary_frame: str = "board_norm",
) -> dict[str, Any]:
    squares = all_square_centers()
    pieces = piece_records(detections) if detections else []
    state = {d["square"]: d["class"] for d in detections}
    frames_available = ["board_norm", "board_uv"]
    board_block: dict[str, Any] = {
        "detected": valid,
        "size_px": BOARD_PX,
        "corners_camera": None,
        "homography_camera_to_board": None,
    }
    if valid and corners_camera is not None:
        board_block["corners_camera"] = corners_camera.reshape(4, 2).tolist()
    if valid and homography is not None:
        board_block["homography_camera_to_board"] = homography.reshape(3, 3).tolist()
    robot = enrich_robot_block(squares, pieces, hand_eye) if valid else None
    if robot is not None:
        frames_available.append(robot["frame"])
    return {
        "schema": SCHEMA_VERSION,
        "seq": seq,
        "frame_id": frame_id,
        "t_host": round(time.time(), 3),
        "valid": valid,
        "frame": primary_frame,
        "frames_available": frames_available,
        "pieces_stale": pieces_stale,
        "board": board_block,
        "squares": squares if valid else {},
        "pieces": pieces,
        "state": state,
        "robot": robot,
    }


@dataclass
class UdpPublisher:
    host: str = "127.0.0.1"
    port: int = 9100
    _seq: int = 0
    _sock: socket.socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def send(self, payload: dict[str, Any]) -> bool:
        self._seq += 1
        payload = dict(payload)
        payload["seq"] = self._seq
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self._sock.sendto(data, (self.host, self.port))
            return True
        except BlockingIOError:
            return False

    def close(self) -> None:
        self._sock.close()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_SQUARE_M = 0.037

DEFAULT_T_TCP_BOARD = np.eye(4, dtype=float)


@dataclass
class HandEyeConfig:
    square_m: float = DEFAULT_SQUARE_M
    T_tcp_board: np.ndarray | None = None
    T_tcp_cam: np.ndarray | None = None

    def to_stream_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"square_m": float(self.square_m)}
        if self.T_tcp_board is not None:
            out["T_tcp_board"] = np.asarray(self.T_tcp_board, dtype=float).reshape(4, 4)
        if self.T_tcp_cam is not None:
            out["T_tcp_cam"] = np.asarray(self.T_tcp_cam, dtype=float).reshape(4, 4)
        return out


def default_hand_eye() -> HandEyeConfig:
    return HandEyeConfig(
        square_m=DEFAULT_SQUARE_M,
        T_tcp_board=DEFAULT_T_TCP_BOARD.copy(),
    )


def resolve_hand_eye(
    *,
    enabled: bool = False,
    square_m: float | None = None,
    T_tcp_board: np.ndarray | None = None,
    T_tcp_cam: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    cfg = default_hand_eye()
    if square_m is not None:
        cfg = HandEyeConfig(
            square_m=float(square_m),
            T_tcp_board=cfg.T_tcp_board,
            T_tcp_cam=cfg.T_tcp_cam,
        )
    if T_tcp_board is not None:
        cfg = HandEyeConfig(
            square_m=cfg.square_m,
            T_tcp_board=np.asarray(T_tcp_board, dtype=float).reshape(4, 4),
            T_tcp_cam=cfg.T_tcp_cam,
        )
    if T_tcp_cam is not None:
        cfg = HandEyeConfig(
            square_m=cfg.square_m,
            T_tcp_board=cfg.T_tcp_board,
            T_tcp_cam=np.asarray(T_tcp_cam, dtype=float).reshape(4, 4),
        )
    return cfg.to_stream_dict()

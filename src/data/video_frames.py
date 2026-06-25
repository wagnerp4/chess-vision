from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2


@dataclass
class ExtractedFrame:
    path: Path
    frame_index: int
    timestamp_sec: float


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 30.0,
    max_seconds: float = 27.0,
) -> List[ExtractedFrame]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0.0:
        native_fps = target_fps

    duration_sec = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / native_fps
    if duration_sec <= 0 and max_seconds <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable duration: {video_path}")

    step = 1.0 / target_fps
    num_frames = int(target_fps * max_seconds)
    extracted: List[ExtractedFrame] = []
    for frame_index in range(num_frames):
        t = frame_index / target_fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        out_path = output_dir / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame: {out_path}")
        extracted.append(
            ExtractedFrame(path=out_path, frame_index=frame_index, timestamp_sec=t)
        )

    cap.release()
    if not extracted:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return extracted

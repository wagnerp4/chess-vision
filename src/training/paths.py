from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    env = os.environ.get("CHESS_VISION_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT


def anchor_path(path_str: str, root: Path | None = None) -> Path:
    base = root or repo_root()
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(config_path: str | Path, root: Path | None = None) -> dict:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = anchor_path(str(path), root)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    req = (requested or "cpu").lower().strip()
    if req.isdigit():
        return req
    if req == "cuda":
        if torch.cuda.is_available():
            return "0"
        req = "mps"
    if req == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            print("MPS not available, using CUDA")
            return "0"
        print("MPS not available, using CPU")
        return "cpu"
    return req

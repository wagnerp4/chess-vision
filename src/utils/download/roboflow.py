from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.training.paths import repo_root

PRESETS_REL = "data/roboflow_presets.yaml"


def load_roboflow_presets(root: Path | None = None) -> dict:
    path = (root or repo_root()) / PRESETS_REL
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {k: v for k, v in raw.items() if isinstance(v, dict) and "workspace" in v}


def list_roboflow_presets(root: Path | None = None) -> list[tuple[str, dict]]:
    presets = load_roboflow_presets(root)
    return [(name, cfg) for name, cfg in sorted(presets.items())]


def _load_env_files(root: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    base = root or repo_root()
    candidates = [
        base / ".env",
        base.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        load_dotenv(path, override=False)


def resolve_roboflow_api_key(root: Path | None = None) -> str:
    _load_env_files(root)
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        base = (root or repo_root()).resolve()
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set.\n"
            "  1. Create a free account at https://app.roboflow.com/\n"
            "  2. Copy your API key from Settings → API\n"
            "  3. export ROBOFLOW_API_KEY='your_key'\n"
            f"     or add ROBOFLOW_API_KEY=... to {base / '.env'}"
        )
    return key


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    *,
    model_format: str = "yolov8",
    overwrite: bool = False,
    api_key: str | None = None,
    root: Path | None = None,
) -> Path:
    from roboflow import Roboflow

    if output_dir.exists() and not overwrite:
        yaml_files = list(output_dir.glob("data.yaml")) + list(output_dir.glob("dataset.yaml"))
        if yaml_files and (
            (output_dir / "train").exists() or any(output_dir.glob("**/images"))
        ):
            print(f"Dataset already present at {output_dir} (use --overwrite to re-download)")
            return output_dir

    key = api_key or resolve_roboflow_api_key(root=root)
    rf = Roboflow(api_key=key)
    print(f"Downloading {workspace}/{project} v{version} as {model_format} → {output_dir}")
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(model_format, location=str(output_dir), overwrite=overwrite)
    )
    location = Path(dataset.location)
    yaml_files = list(location.glob("data.yaml")) + list(location.glob("dataset.yaml"))
    print(f"Download complete: {location}")
    if yaml_files:
        print(f"Dataset yaml: {yaml_files[0]}")
    else:
        print("Warning: no data.yaml found in download directory")
    return location

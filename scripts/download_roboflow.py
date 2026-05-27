import argparse
import os
import sys
from pathlib import Path

import yaml

VISION_ROOT = Path(__file__).resolve().parent.parent
PRESETS_PATH = VISION_ROOT / "configs" / "roboflow_datasets.yaml"


def load_presets() -> dict:
    with open(PRESETS_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return {k: v for k, v in raw.items() if isinstance(v, dict) and "workspace" in v}


def resolve_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(VISION_ROOT.parent / ".env")
        load_dotenv(VISION_ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set.\n"
            "  1. Create a free account at https://app.roboflow.com/\n"
            "  2. Copy your API key from Settings → API\n"
            "  3. export ROBOFLOW_API_KEY='your_key'\n"
            "     or add it to .env in the repo root"
        )
    return key


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    model_format: str = "yolov8",
    overwrite: bool = False,
) -> Path:
    from roboflow import Roboflow

    if output_dir.exists() and not overwrite:
        yaml_files = list(output_dir.glob("data.yaml")) + list(output_dir.glob("dataset.yaml"))
        if yaml_files and any((output_dir / "train").exists() or output_dir.glob("**/images")):
            print(f"Dataset already present at {output_dir} (use --overwrite to re-download)")
            return output_dir

    api_key = resolve_api_key()
    rf = Roboflow(api_key=api_key)
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


def main():
    presets = load_presets()
    parser = argparse.ArgumentParser(
        description="Download a Roboflow Universe chess dataset (YOLOv8 format, no browser)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(presets.keys()),
        default="chess-pieces-2",
        help="Named dataset from configs/roboflow_datasets.yaml",
    )
    parser.add_argument("--workspace", type=str, default=None, help="Override Roboflow workspace slug")
    parser.add_argument("--project", type=str, default=None, help="Override Roboflow project slug")
    parser.add_argument("--version", type=int, default=None, help="Override dataset version number")
    parser.add_argument(
        "--format",
        type=str,
        default="yolov8",
        help="Export format (yolov8, coco, voc, ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/roboflow/<preset>/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if directory exists")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, cfg in presets.items():
            desc = cfg.get("description", "")
            print(
                f"  {name}: {cfg['workspace']}/{cfg['project']} v{cfg['version']}"
                + (f"  — {desc}" if desc else "")
            )
        return

    preset_cfg = presets[args.preset]
    workspace = args.workspace or preset_cfg["workspace"]
    project = args.project or preset_cfg["project"]
    version = args.version if args.version is not None else int(preset_cfg["version"])
    output_dir = Path(args.output) if args.output else VISION_ROOT / "data" / "roboflow" / args.preset
    if not output_dir.is_absolute():
        output_dir = VISION_ROOT / output_dir

    try:
        download_roboflow_dataset(
            workspace=workspace,
            project=project,
            version=version,
            output_dir=output_dir,
            model_format=args.format,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "\nIf workspace/project/version are wrong, check the Universe URL and pass overrides:\n"
            "  --workspace <slug> --project <slug> --version <n>",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

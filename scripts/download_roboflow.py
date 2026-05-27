import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.utils.download.roboflow import (
    download_roboflow_dataset,
    list_roboflow_presets,
    load_roboflow_presets,
)


def main() -> None:
    presets = load_roboflow_presets(VISION_ROOT)
    parser = argparse.ArgumentParser(
        description="Download a Roboflow Universe chess dataset (YOLOv8 format, no browser)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(presets.keys()),
        default="chess-pieces-2",
        help="Named dataset from data/roboflow_presets.yaml",
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
        for name, cfg in list_roboflow_presets(VISION_ROOT):
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
            root=VISION_ROOT,
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

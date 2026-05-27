import argparse
from pathlib import Path
import sys

VISION_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(VISION_ROOT))

from src.data.class_mapping import normalize_dataset_tree


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Roboflow YOLO export to canonical lowercase 12-class layout"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Roboflow download directory (contains data.yaml and train/valid/test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/processed/<input_folder_name>)",
    )
    parser.add_argument(
        "--no-legacy-kaggle-map",
        action="store_true",
        help="Disable Kaggle-style piece name aliases (camel/elephant/horse)",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink images instead of copying (saves disk space)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = VISION_ROOT / input_dir
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = VISION_ROOT / output_dir
    else:
        output_dir = VISION_ROOT / "data" / "processed" / input_dir.name

    legacy = not args.no_legacy_kaggle_map
    yaml_path = normalize_dataset_tree(
        input_dir=input_dir,
        output_dir=output_dir,
        legacy_kaggle_map=legacy,
        symlink_images=args.symlink_images,
    )
    print(f"Normalized dataset written to: {output_dir}")
    print(f"Training yaml: {yaml_path}")


if __name__ == "__main__":
    main()

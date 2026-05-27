import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.utils.download.kaggle import download_kaggle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download chess pieces dataset from Kaggle")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ninadaithal/chess-pieces-dataset",
        help="Kaggle dataset name (format: username/dataset-name)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kaggle",
        help="Output directory for downloaded dataset",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = VISION_ROOT / output

    try:
        download_kaggle_dataset(args.dataset, output)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.kaggle.com/datasets/{args.dataset}")
        print(f"Then extract to: {output}")
        sys.exit(1)


if __name__ == "__main__":
    main()

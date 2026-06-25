import argparse
import sys
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.utils.visualization import plot_random_dataset_samples

DEFAULT_DATASET = VISION_ROOT / "data" / "processed" / "neuroTUM-chess-dataset"
DEFAULT_OUTPUT = VISION_ROOT / "outputs" / "nt_dataset_debug"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot random YOLO-labeled dataset samples with bounding boxes"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_DATASET / "test" / "images"),
        help="Directory of dataset images",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=str(DEFAULT_DATASET / "test" / "labels"),
        help="Directory of YOLO label files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Where to save annotated debug images",
    )
    parser.add_argument("--n", type=int, default=4, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    saved = plot_random_dataset_samples(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        n=args.n,
        seed=args.seed,
    )
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
import argparse
from pathlib import Path
import zipfile

# TODO: extend download util
# think about moving it to src long-term?

def download_kaggle_dataset(dataset_name: str, output_dir: str):
    """Download dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API not installed. Please install: pip install kaggle\n"
            "Then set up your Kaggle API credentials: "
            "https://www.kaggle.com/docs/api"
        )
    
    api = KaggleApi()
    api.authenticate()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_path}")
    
    api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True
    )
    
    print(f"Dataset downloaded successfully to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download chess pieces dataset from Kaggle")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ninadaithal/chess-pieces-dataset",
        help="Kaggle dataset name (format: username/dataset-name)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded dataset"
    )
    
    args = parser.parse_args()
    
    try:
        download_kaggle_dataset(args.dataset, args.output)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.kaggle.com/datasets/{args.dataset}")
        print(f"Then extract to: {args.output}")


if __name__ == "__main__":
    main()


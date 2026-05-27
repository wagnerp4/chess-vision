from __future__ import annotations

from pathlib import Path


def download_kaggle_dataset(dataset_name: str, output_dir: str | Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "Kaggle API not installed. Please install: pip install kaggle\n"
            "Then set up your Kaggle API credentials: "
            "https://www.kaggle.com/docs/api"
        ) from exc

    api = KaggleApi()
    api.authenticate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_path}")

    api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True,
    )

    print(f"Dataset downloaded successfully to {output_path}")
    return output_path

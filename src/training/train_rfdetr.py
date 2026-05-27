import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_rfdetr(config: dict):
    try:
        from roboflow import Roboflow
        from rfdetr import (
            RFDETRBase,
            RFDETRLarge,
            RFDETRMedium,
            RFDETRNano,
            RFDETRSmall
        )
    except ImportError:
        raise ImportError(
            "RF-DETR dependencies not installed. "
            "Please install: pip install roboflow rfdetr"
        )
    
    dataset_path = config["data"]["dataset_path"]
    if not Path(dataset_path).is_absolute():
        dataset_path = Path.cwd() / dataset_path

    dataset_yaml = Path(dataset_path)
    if dataset_yaml.is_dir():
        dataset_yaml = dataset_yaml / "dataset.yaml"
    elif dataset_yaml.suffix in (".yaml", ".yml"):
        pass
    else:
        dataset_yaml = dataset_yaml / "dataset.yaml"
    
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    model_size = config.get("model", {}).get("size", "base").lower()
    model_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge
    }
    
    if model_size not in model_map:
        raise ValueError(
            f"Invalid model size: {model_size}. "
            f"Must be one of: {list(model_map.keys())}"
        )
    
    model_class = model_map[model_size]
    model = model_class()
    training_params = {
        "data": str(dataset_yaml),
        "epochs": config["training"]["epochs"],
        "batch_size": config["data"]["batch_size"],
        "learning_rate": config["training"]["optimizer"]["lr"],
        "weight_decay": config["training"]["optimizer"]["weight_decay"],
        "warmup_epochs": config["training"]["optimizer"]["warmup_epochs"],
        "device": config["training"]["device"],
        "output_dir": config["training"]["save_dir"],
    }
    
    results = model.train(**training_params)
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = Path(results.save_dir) / "best.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, save_dir / "best.pt")
        print(f"Best model saved to {save_dir / 'best.pt'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR model for chess piece detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rfdetr.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    try:
        results = train_rfdetr(config)
        print("Training completed!")
        print(f"Results saved to: {results.save_dir}")
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nNote: RF-DETR training may require additional setup.")
        print("Please refer to RF-DETR documentation for detailed instructions.")


if __name__ == "__main__":
    main()


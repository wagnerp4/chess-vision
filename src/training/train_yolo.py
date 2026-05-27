import argparse
import shutil
import yaml
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

VISION_ROOT = Path(__file__).resolve().parent.parent


def anchor_vision(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (VISION_ROOT / p).resolve()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
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
        print("MPS not available, using CPU")
        return "cpu"
    return req


def build_training_params(config: dict) -> dict:
    dataset_yaml = anchor_vision(config["data"]["dataset_path"])
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    training = config["training"]
    device = resolve_device(training["device"])
    aug_params = training["augmentation"]

    params = {
        "data": str(dataset_yaml),
        "epochs": training["epochs"],
        "patience": training["patience"],
        "device": device,
        "project": str(anchor_vision(training["project"])),
        "name": training["name"],
        "batch": config["data"]["batch_size"],
        "workers": config["data"]["num_workers"],
        "imgsz": config["model"]["input_size"],
        "plots": training.get("plots", False),
        "amp": training.get("amp", False),
        "lr0": training["optimizer"]["lr0"],
        "lrf": training["optimizer"]["lrf"],
        "momentum": training["optimizer"]["momentum"],
        "weight_decay": training["optimizer"]["weight_decay"],
        "warmup_epochs": training["optimizer"]["warmup_epochs"],
        "warmup_momentum": training["optimizer"]["warmup_momentum"],
        "warmup_bias_lr": training["optimizer"]["warmup_bias_lr"],
        "hsv_h": aug_params["hsv_h"],
        "hsv_s": aug_params["hsv_s"],
        "hsv_v": aug_params["hsv_v"],
        "degrees": aug_params["degrees"],
        "translate": aug_params["translate"],
        "scale": aug_params["scale"],
        "shear": aug_params["shear"],
        "perspective": aug_params["perspective"],
        "flipud": aug_params["flipud"],
        "fliplr": aug_params["fliplr"],
        "mosaic": aug_params["mosaic"],
        "mixup": aug_params["mixup"],
    }
    print(f"Training device: {device}  plots={params['plots']}  amp={params['amp']}")
    return params


def archive_best_checkpoint(best_model_path: Path, config: dict) -> None:
    save_dir = anchor_vision(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, save_dir / "best.pt")
    print(f"Best model saved to {save_dir / 'best.pt'}")

    best_checkpoints_dir = anchor_vision("data/checkpoints/best")
    best_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    max_size_mb = 50.0
    run_name = config["training"]["name"]

    if file_size_mb <= max_size_mb:
        dated = datetime.now().strftime("%Y%m%d")
        versioned_path = best_checkpoints_dir / f"best_{dated}_{run_name}.pt"
        shutil.copy(best_model_path, versioned_path)
        print(f"Best checkpoint archived to {versioned_path} ({file_size_mb:.2f} MB)")
    else:
        print(
            f"Best checkpoint ({file_size_mb:.2f} MB) exceeds size limit ({max_size_mb} MB), not archiving"
        )


def train_yolo(config: dict):
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]

    pretrained_dir = anchor_vision("data/checkpoints/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    if pretrained:
        pretrained_path = pretrained_dir / f"{model_name}.pt"
        root_model_path = Path(f"{model_name}.pt")

        if pretrained_path.exists():
            model = YOLO(str(pretrained_path))
        elif root_model_path.exists():
            shutil.move(str(root_model_path), str(pretrained_path))
            model = YOLO(str(pretrained_path))
            print(f"Pretrained model moved to {pretrained_path}")
        else:
            model = YOLO(model_name)
            if root_model_path.exists():
                shutil.move(str(root_model_path), str(pretrained_path))
                print(f"Pretrained model downloaded and moved to {pretrained_path}")
    else:
        model = YOLO(f"{model_name}.yaml")

    results = model.train(**build_training_params(config))

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        archive_best_checkpoint(best_model_path, config)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for chess piece detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/yolo.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt in the Ultralytics run dir (not best.pt)",
    )

    args = parser.parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = VISION_ROOT / cfg_path
    config = load_config(str(cfg_path))

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model = YOLO(str(resume_path))
        params = build_training_params(config)
        params["resume"] = True
        results = model.train(**params)
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            archive_best_checkpoint(best_model_path, config)
    else:
        results = train_yolo(config)

    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()

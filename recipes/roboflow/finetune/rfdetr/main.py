import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
os.chdir(root)
sys.path.insert(0, str(root))

from recipes.roboflow.finetune.setting import DEFAULT_CONFIG_RFDETR, prepare_run
from recipes.roboflow.finetune.rfdetr.train import train_rfdetr


def main() -> None:
    _args, config, root = prepare_run(DEFAULT_CONFIG_RFDETR)
    try:
        results = train_rfdetr(config, root)
        print("Training completed!")
        print(f"Results saved to: {results.save_dir}")
    except Exception as exc:
        print(f"Error during training: {exc}")
        print("\nNote: RF-DETR training may require additional setup.")
        print("Please refer to RF-DETR documentation for detailed instructions.")
        raise


if __name__ == "__main__":
    main()

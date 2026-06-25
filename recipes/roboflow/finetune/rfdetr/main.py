import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
os.chdir(root)
sys.path.insert(0, str(root))

from recipes.roboflow.finetune.setting import DEFAULT_CONFIG_RFDETR, prepare_run
from recipes.roboflow.finetune.rfdetr.train import train_rfdetr
from src.training.run_mode import parse_run_mode


def main() -> None:
    args, config, root = prepare_run(DEFAULT_CONFIG_RFDETR, allow_run_modes=True)
    run_mode = parse_run_mode(args)
    try:
        results = train_rfdetr(config, root, run_mode=run_mode)
        print("Training completed!")
        print(f"Results saved to: {results.output_dir}")
        if results.archived_checkpoint:
            print(f"Archived checkpoint: {results.archived_checkpoint}")
    except Exception as exc:
        print(f"Error during training: {exc}")
        print("\nNote: RF-DETR training may require additional setup.")
        print("Please refer to RF-DETR documentation for detailed instructions.")
        raise


if __name__ == "__main__":
    main()

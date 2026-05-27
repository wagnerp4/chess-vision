import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
os.chdir(root)
sys.path.insert(0, str(root))

from recipes.roboflow.finetune.setting import DEFAULT_CONFIG_YOLO, prepare_run
from recipes.roboflow.finetune.yolo.train import resume_yolo, train_yolo


def main() -> None:
    args, config, root = prepare_run(DEFAULT_CONFIG_YOLO, allow_resume=True)

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        results = resume_yolo(config, resume_path, root)
    else:
        results = train_yolo(config, root)

    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()

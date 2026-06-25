import argparse
import sys
from datetime import datetime
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.training.rfdetr_checkpoints import export_rfdetr_checkpoint, resolve_rfdetr_source


def default_rfdetr_run_dir() -> Path:
    return VISION_ROOT / "data" / "checkpoints" / "rfdetr" / "rfdetr_large_chess_pieces_2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR training checkpoint to portable best_YYYYMMDD_rfdetr.pt"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to run dir, .pth, or .ckpt (default: data/checkpoints/rfdetr/rfdetr_large_chess_pieces_2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path (default: data/checkpoints/best/best_YYYYMMDD_rfdetr.pt)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date stamp YYYYMMDD for default output name (default: today)",
    )
    args = parser.parse_args()

    source = args.source or str(default_rfdetr_run_dir())
    source_path = resolve_rfdetr_source(source if Path(source).is_absolute() else VISION_ROOT / source)
    print(f"Source: {source_path}")

    output = args.output
    if output is not None and not Path(output).is_absolute():
        output = str(VISION_ROOT / output)

    dated = args.date or datetime.now().strftime("%Y%m%d")
    out_path = export_rfdetr_checkpoint(
        source_path,
        output_path=output,
        root=VISION_ROOT,
        date=dated,
    )
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

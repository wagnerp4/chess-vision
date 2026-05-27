import argparse
import pickle
import shutil
import sys

if sys.platform != "win32":
    import pathlib
    from pathlib import Path
    
    pathlib.WindowsPath = Path
    sys.modules["pathlib"].WindowsPath = Path
else:
    from pathlib import Path

from ultralytics import YOLO

_VISION_ROOT = Path(__file__).resolve().parent.parent


def default_yolo_checkpoint() -> Path:
    for p in (
        _VISION_ROOT / "data" / "runs" / "detect" / "yolo_chess_pieces_2" / "weights" / "best.pt",
        _VISION_ROOT / "data" / "checkpoints" / "best" / "best_20260515_yolo_chess_pieces_2.pt",
        _VISION_ROOT / "data" / "checkpoints" / "yolo" / "best.pt",
    ):
        if p.is_file():
            return p
    return _VISION_ROOT / "data" / "checkpoints" / "best" / "best_20260110_122310.pt"


def convert_to_oak_blob(
    model_path: str,
    output_dir: str = "data/checkpoints/oak",
    input_size: int = 640,
):
    """
    Convert YOLO model to OAK .blob format.
    
    Args:
        model_path: Path to trained .pt model
        output_dir: Directory to save converted model
        input_size: Input image size (default 640)
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    onnx_path = output_dir / f"{model_path.stem}.onnx"
    blob_path = output_dir / f"{model_path.stem}.blob"
    
    print(f"Exporting to ONNX format...")
    try:
        model.export(
            format="onnx",
            imgsz=input_size,
            simplify=True,
            opset=12,
        )
        
        exported_onnx = model_path.parent / f"{model_path.stem}.onnx"
        if not exported_onnx.exists():
            exported_onnx = Path(f"{model_path.stem}.onnx")
        
        if exported_onnx.exists():
            shutil.move(str(exported_onnx), str(onnx_path))
            print(f"ONNX model saved to {onnx_path}")
        else:
            raise FileNotFoundError("ONNX export failed - file not created")
            
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        raise
    
    print(f"\nConverting ONNX to .blob format...")
    print("Note: This requires blobconverter. Install with: pip install blobconverter")
    
    try:
        import blobconverter
    except ImportError:
        print("\nblobconverter not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "blobconverter"])
        import blobconverter
    
    try:
        blob_path = blobconverter.from_onnx(
            model=str(onnx_path),
            data_type="FP16",
            shaves=6,
        )
        
        if blob_path and Path(blob_path).exists():
            final_blob = output_dir / f"{model_path.stem}.blob"
            shutil.copy(str(blob_path), str(final_blob))
            print(f"\nConversion successful!")
            print(f"Blob model saved to: {final_blob}")
            print(f"\nTo use in OAK Viewer:")
            print(f"1. Open OAK Viewer")
            print(f"2. Click 'Swap Model' button in the pipeline view")
            print(f"3. Select the .blob file: {final_blob}")
            return str(final_blob)
        else:
            raise FileNotFoundError("Blob conversion failed - file not created")
    except Exception as e:
        print(f"\nError during blob conversion: {e}")
        print(f"\nAlternative: Use online converter at https://blobconverter.luxonis.com/")
        print(f"Upload your ONNX file: {onnx_path}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO model to OAK .blob format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained .pt model (default: latest chess-pieces-2 or yolo/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/checkpoints/oak",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else default_yolo_checkpoint()
    if not model_path.is_absolute():
        model_path = _VISION_ROOT / model_path
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = _VISION_ROOT / out_dir

    convert_to_oak_blob(
        model_path=str(model_path),
        output_dir=str(out_dir),
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()

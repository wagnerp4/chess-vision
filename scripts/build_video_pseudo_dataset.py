import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

VISION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_ROOT))

from src.data.class_mapping import write_dataset_yaml
from src.data.roboflow_inference import (
    DEFAULT_API_URL,
    DEFAULT_MODEL_ID,
    build_inference_client,
    extract_image_size,
    extract_predictions,
    infer_image_cached,
    predictions_to_label_file,
)
from src.data.video_frames import ExtractedFrame, extract_frames
from src.utils.download.roboflow import resolve_roboflow_api_key

DEFAULT_VIDEO = VISION_ROOT / "data" / "videos" / "neuroTUM-chess-dataset.mp4"
DEFAULT_OUTPUT = VISION_ROOT / "data" / "processed" / "neuroTUM-chess-dataset"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract video frames and pseudo-label them via Roboflow Inference API"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO),
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output processed dataset directory",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Roboflow model id (project/version)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="Roboflow inference server URL",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target extraction fps")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=27.0,
        help="Maximum video duration to sample (seconds)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence for pseudo labels",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing test/images frames (skip video extraction)",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only extract frames; do not call Roboflow API",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = VISION_ROOT / video_path
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = VISION_ROOT / output_root

    images_dir = output_root / "test" / "images"
    labels_dir = output_root / "test" / "labels"
    cache_dir = output_root / "cache" / "predictions"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        frame_paths = sorted(images_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames in {images_dir}; run without --skip-extract")
        frames = [
            ExtractedFrame(path=p, frame_index=i, timestamp_sec=i / args.fps)
            for i, p in enumerate(frame_paths)
        ]
    else:
        print(f"Extracting frames from {video_path} at {args.fps} fps for {args.max_seconds}s")
        frames = extract_frames(
            video_path=video_path,
            output_dir=images_dir,
            target_fps=args.fps,
            max_seconds=args.max_seconds,
        )
        print(f"Extracted {len(frames)} frames to {images_dir}")

    if args.skip_infer:
        yaml_path = write_dataset_yaml(output_root, split_keys=["test"])
        print(f"Skipped inference. Dataset yaml: {yaml_path}")
        return 0

    api_key = resolve_roboflow_api_key(root=VISION_ROOT)
    client = build_inference_client(api_key=api_key, api_url=args.api_url)

    unmapped_classes: set[str] = set()
    total_boxes = 0
    manifest_entries = []
    cache_hits = 0
    api_calls = 0

    for frame in tqdm(frames, desc="Roboflow inference"):
        img_path = frame.path
        stem = img_path.stem
        cache_path = cache_dir / f"{stem}.json"
        had_cache = cache_path.is_file()

        result = infer_image_cached(
            client=client,
            image_path=img_path,
            cache_path=cache_path,
            model_id=args.model_id,
            confidence=args.confidence,
        )
        if had_cache:
            cache_hits += 1
        else:
            api_calls += 1

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read frame: {img_path}")
        fallback_h, fallback_w = img.shape[:2]
        img_w, img_h = extract_image_size(result, fallback_w, fallback_h)
        predictions = extract_predictions(result)
        label_path = labels_dir / f"{stem}.txt"
        kept = predictions_to_label_file(
            predictions=predictions,
            img_w=img_w,
            img_h=img_h,
            out_path=label_path,
            min_confidence=args.confidence,
            unmapped_classes=unmapped_classes,
        )
        total_boxes += kept
        manifest_entries.append(
            {
                "stem": stem,
                "frame_index": frame.frame_index,
                "timestamp_sec": round(frame.timestamp_sec, 4),
                "num_boxes": kept,
                "image": str(img_path.relative_to(output_root)),
                "label": str(label_path.relative_to(output_root)),
            }
        )

    yaml_path = write_dataset_yaml(output_root, split_keys=["test"])
    manifest_path = output_root / "manifest.json"
    manifest = {
        "video": str(video_path.resolve()),
        "model_id": args.model_id,
        "fps": args.fps,
        "max_seconds": args.max_seconds,
        "confidence": args.confidence,
        "num_frames": len(frames),
        "total_boxes": total_boxes,
        "cache_hits": cache_hits,
        "api_calls": api_calls,
        "unmapped_classes": sorted(unmapped_classes),
        "frames": manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset written to {output_root}")
    print(f"Training yaml: {yaml_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Frames: {len(frames)}, boxes: {total_boxes}, API calls: {api_calls}, cache hits: {cache_hits}")
    if unmapped_classes:
        print(f"Unmapped classes (dropped): {sorted(unmapped_classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

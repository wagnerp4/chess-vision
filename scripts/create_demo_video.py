"""
Script to create a demo video showing chess piece detection on multiple frames.
Uses YOLOv11 model from Hugging Face (dopaul/chessboard-detector) and Kaggle dataset.
"""
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import random
import tempfile


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


def find_image_files(dataset_dir: Path) -> List[Path]:
    """Find all image files in the dataset directory."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(dataset_dir.rglob(f"*{ext}"))
        image_files.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def draw_detections(image: np.ndarray, results, class_names: dict, conf_threshold: float = 0.1) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores on image."""
    annotated_image = image.copy()
    
    if results.boxes is None or len(results.boxes) == 0:
        return annotated_image
    
    boxes = results.boxes
    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        
        if conf < conf_threshold:
            continue
        
        xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
        x1, y1, x2, y2 = map(int, xyxy)
        
        class_name = class_names.get(cls, f"class_{cls}")
        label = f"{class_name}: {conf:.2f}"
        
        color = (0, 255, 0)
        thickness = 2
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        label_y = max(y1, text_height + 10)
        cv2.rectangle(
            annotated_image,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 10, label_y),
            color,
            -1
        )
        cv2.putText(
            annotated_image,
            label,
            (x1 + 5, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return annotated_image


def add_frame_info(image: np.ndarray, frame_num: int, total_frames: int, num_detections: int) -> np.ndarray:
    """Add frame information text to the image."""
    info_text = f"Frame {frame_num}/{total_frames} | Detections: {num_detections}"
    
    cv2.putText(
        image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1
    )
    
    return image


def download_model_from_huggingface(repo_id: str, cache_dir: Optional[str] = None) -> str:
    """Download YOLO model from Hugging Face and return local path."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    print(f"Downloading model from Hugging Face: {repo_id}")
    
    try:
        repo_files = list_repo_files(repo_id, repo_type="model")
        pt_files = [f for f in repo_files if f.endswith(".pt")]
        
        if not pt_files:
            raise Exception(f"No .pt files found in repository {repo_id}")
        
        print(f"Found model files: {pt_files}")
        model_filename = pt_files[0]
        
        if len(pt_files) > 1:
            preferred = [f for f in pt_files if any(x in f.lower() for x in ["best", "model", "weights", "yolo"])]
            if preferred:
                model_filename = preferred[0]
        
        print(f"Downloading: {model_filename}")
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            cache_dir=cache_dir
        )
        print(f"Model downloaded to: {model_file}")
        return model_file
    except Exception as e:
        print(f"Error listing/downloading from repo, trying common filenames...")
        for filename in ["model.pt", "best.pt", "weights.pt", "yolo.pt", "checkpoint.pt"]:
            try:
                model_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir
                )
                print(f"Model downloaded to: {model_file}")
                return model_file
            except:
                continue
        
        raise Exception(f"Could not find model file in repository {repo_id}. Error: {e}")


def load_yolo_model(model_identifier: str) -> YOLO:
    """Load YOLO model from various sources (local path, Hugging Face, or pretrained)."""
    model_path = model_identifier
    
    if "/" in model_identifier and not os.path.exists(model_identifier):
        if not model_identifier.endswith(".pt") and not model_identifier.endswith(".yaml"):
            print(f"Detected Hugging Face model ID: {model_identifier}")
            model_path = download_model_from_huggingface(model_identifier)
    
    if not os.path.exists(model_path) and not model_path.endswith(".yaml"):
        print(f"Model path not found: {model_path}")
        print("Trying to use as pretrained model name (e.g., yolov8n, yolov11n)...")
        try:
            model = YOLO(model_path)
            return model
        except:
            print(f"Could not load {model_path} as pretrained model")
            print("Available options:")
            print("  - Local .pt file path")
            print("  - Hugging Face repo (username/repo-name)")
            print("  - Pretrained model name (yolov8n, yolov11n, etc.)")
            raise
    
    return YOLO(model_path)


def create_demo_video(
    model_name: str = "dopaul/chessboard-detector",
    dataset_dir: str = "data/raw",
    output_video: str = "demo_chess_detection.mp4",
    num_frames: int = 10,
    fps: int = 2,
    conf_threshold: float = 0.1,
    download_dataset: bool = False,
    dataset_name: str = "ninadaithal/chess-pieces-dataset"
):
    """Create a demo video with chess piece detections."""
    
    print("Loading YOLO model...")
    model_loaded = False
    try:
        model = load_yolo_model(model_name)
        print(f"Model loaded successfully")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying fallback: using pretrained YOLOv8n model...")
        try:
            model = YOLO("yolov8n.pt")
            print("Using pretrained YOLOv8n (general object detection)")
            print("WARNING: This model is trained on COCO dataset and may not detect chess pieces well.")
            print("Consider training a chess-specific model or using a chess-trained checkpoint.")
            model_loaded = True
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Make sure ultralytics is installed: pip install ultralytics")
            print("For Hugging Face models, also install: pip install huggingface_hub")
            return
    
    if model_loaded:
        print(f"Model classes: {list(model.names.values())[:10]}...")
        print(f"Total classes: {len(model.names)}")
        print(f"Using confidence threshold: {conf_threshold}")
    
    dataset_path = Path(dataset_dir)
    
    if download_dataset or not dataset_path.exists():
        print("Downloading dataset from Kaggle...")
        try:
            download_kaggle_dataset(dataset_name, str(dataset_path))
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure Kaggle API credentials are set up.")
            print("Alternatively, place chess images in the dataset directory manually.")
            return
    
    print(f"Searching for images in: {dataset_path}")
    image_files = find_image_files(dataset_path)
    
    if len(image_files) == 0:
        print(f"No image files found in {dataset_path}")
        print("Please ensure the dataset directory contains image files.")
        return
    
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) < num_frames:
        print(f"Warning: Only {len(image_files)} images available, using all of them")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_frames)
    
    print(f"Processing {len(selected_images)} frames...")
    
    frames = []
    class_names = model.names
    
    for idx, image_path in enumerate(selected_images, 1):
        print(f"Processing frame {idx}/{len(selected_images)}: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        results = model.predict(
            image,
            conf=conf_threshold,
            verbose=False
        )
        
        num_detections = 0
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            num_detections = len(results[0].boxes)
            print(f"  Detections: {num_detections} boxes")
            if num_detections > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                print(f"  Confidence range: {confs.min():.3f} - {confs.max():.3f}")
                classes = [class_names[int(cls)] for cls in results[0].boxes.cls[:min(5, num_detections)]]
                print(f"  Sample classes: {classes}")
        else:
            print(f"  No detections found with conf >= {conf_threshold}")
            print(f"  Trying with lower confidence threshold (0.01) to check if model detects anything...")
            results_low_conf = model.predict(
                image,
                conf=0.01,
                verbose=False
            )
            if results_low_conf[0].boxes is not None and len(results_low_conf[0].boxes) > 0:
                low_conf_count = len(results_low_conf[0].boxes)
                print(f"  Found {low_conf_count} detections with conf >= 0.01")
                print(f"  Consider using --conf 0.01 or lower (current: {conf_threshold})")
                results = results_low_conf
                num_detections = low_conf_count
        
        if num_detections > 0:
            annotated_image = draw_detections(image, results[0], class_names, conf_threshold)
        else:
            annotated_image = image.copy()
            print(f"  WARNING: No detections. The model may not be trained for chess pieces.")
            print(f"  Consider using a chess-specific model or training your own.")
            print(f"  You can train a model using: python src/training/train_yolo.py --config configs/yolo_config.yaml")
        
        annotated_image = add_frame_info(annotated_image, idx, len(selected_images), num_detections)
        
        frames.append(annotated_image)
    
    if len(frames) == 0:
        print("No frames processed successfully")
        return
    
    print(f"Creating video with {len(frames)} frames...")
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    if not os.path.exists(os.path.dirname(output_video)) and os.path.dirname(output_video):
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Warning: XVID codec not available, trying H264...")
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print("Warning: H264 codec not available, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    
    print(f"Demo video created successfully: {output_video}")
    print(f"Video properties: {len(frames)} frames, {fps} FPS, {width}x{height} resolution")


def main():
    parser = argparse.ArgumentParser(
        description="Create a demo video showing chess piece detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dopaul/chessboard-detector",
        help="YOLO model name from Hugging Face (default: dopaul/chessboard-detector)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw",
        help="Directory containing chess images (default: data/raw)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_chess_detection.mp4",
        help="Output video file path (default: demo_chess_detection.mp4)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to process (default: 10)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for output video (default: 2)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold for detections (default: 0.1)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle if not present"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ninadaithal/chess-pieces-dataset",
        help="Kaggle dataset name (default: ninadaithal/chess-pieces-dataset)"
    )
    
    args = parser.parse_args()
    
    create_demo_video(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        output_video=args.output,
        num_frames=args.num_frames,
        fps=args.fps,
        conf_threshold=args.conf,
        download_dataset=args.download,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()


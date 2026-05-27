"""
Inference script for RF-DETR model.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from src.models.backends.rfdetr import build_rfdetr


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_detections(image, detections, class_names):
    """Draw bounding boxes and labels on image."""
    for det in detections:
        bbox = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        label = f"{cls}: {conf:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return image


def infer_rfdetr(
    model_path: str,
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    model_size: str = "base"
):
    """Run inference on image using RF-DETR model."""
    model = build_rfdetr(model_size, checkpoint=model_path)
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    results = model.predict(
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    detections = []
    for result in results:
        detections.append({
            "class": result["class"],
            "confidence": result["confidence"],
            "bbox": result["bbox"]
        })
    
    annotated_image = draw_detections(image.copy(), detections, {})
    
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Output saved to: {output_path}")
    else:
        cv2.imshow("Detection Results", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections, annotated_image


def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on chess piece images")
    parser.add_argument("--model", type=str, required=True, help="Path to RF-DETR model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to config file for thresholds")
    parser.add_argument("--model-size", type=str, default="base", help="Model size: nano, small, base, medium, large")
    
    args = parser.parse_args()
    
    conf_threshold = args.conf
    iou_threshold = args.iou
    model_size = args.model_size
    
    if args.config:
        config = load_config(args.config)
        conf_threshold = config["evaluation"]["conf_threshold"]
        iou_threshold = config["evaluation"]["iou_threshold"]
        model_size = config.get("model", {}).get("size", model_size)
    
    try:
        detections, annotated_image = infer_rfdetr(
            args.model,
            args.image,
            args.output,
            conf_threshold,
            iou_threshold,
            model_size
        )
        
        print(f"\nDetected {len(detections)} chess pieces:")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class']}: {det['confidence']:.2f}")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("\nNote: RF-DETR inference may require additional setup.")
        print("Please refer to RF-DETR documentation for detailed instructions.")


if __name__ == "__main__":
    main()


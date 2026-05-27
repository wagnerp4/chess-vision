"""
Inference script for YOLO models.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_detections(image, detections, class_names):
    """Draw bounding boxes and labels on image."""
    for det in detections:
        box = det.boxes
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        class_name = class_names.get(cls, f"class_{cls}")
        label = f"{class_name}: {conf:.2f}"
        
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


def infer_yolo(
    model_path: str,
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """Run inference on image using YOLO model."""
    model = YOLO(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    class_names = model.names
    
    annotated_image = draw_detections(image.copy(), results[0], class_names)
    
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Output saved to: {output_path}")
    else:
        cv2.imshow("Detection Results", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    detections = []
    for det in results[0]:
        box = det.boxes
        for i in range(len(box)):
            detections.append({
                "class": class_names[int(box.cls[i])],
                "confidence": float(box.conf[i]),
                "bbox": box.xyxy[i].cpu().numpy().tolist()
            })
    
    return detections, annotated_image


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on chess piece images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--config", type=str, default=None, help="Path to config file for thresholds")
    
    args = parser.parse_args()
    
    conf_threshold = args.conf
    iou_threshold = args.iou
    
    if args.config:
        config = load_config(args.config)
        conf_threshold = config["evaluation"]["conf_threshold"]
        iou_threshold = config["evaluation"]["iou_threshold"]
    
    detections, annotated_image = infer_yolo(
        args.model,
        args.image,
        args.output,
        conf_threshold,
        iou_threshold
    )
    
    print(f"\nDetected {len(detections)} chess pieces:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class']}: {det['confidence']:.2f}")


if __name__ == "__main__":
    main()


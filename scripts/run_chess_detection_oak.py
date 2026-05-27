import argparse
import cv2
import depthai as dai
import numpy as np
from pathlib import Path


CHESS_CLASSES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
COLORS = [
    (255, 0, 0),      # Red - pawn
    (0, 255, 0),      # Green - rook
    (0, 0, 255),      # Blue - knight
    (255, 255, 0),    # Cyan - bishop
    (255, 0, 255),    # Magenta - queen
    (0, 255, 255),    # Yellow - king
]


def create_chess_detection_pipeline(blob_path: str, input_size: int = 640):
    
    pipeline = dai.Pipeline()
    
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.setSensorType(dai.CameraSensorType.COLOR)
    
    cam_output = cam_rgb.requestOutput((input_size, input_size))
    
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    
    cam_output.link(nn.input)
    
    return pipeline, cam_rgb, nn


def decode_yolo_output(detections, frame_shape, input_size=640):
    """
    Decode YOLO neural network output to bounding boxes.
    
    YOLO models output format: [batch, num_detections, 4+1+num_classes]
    Where: 4 (bbox coordinates) + 1 (objectness) + num_classes (class scores)
    
    Args:
        detections: NNData output from neural network
        frame_shape: (height, width) of the frame
        input_size: Input size used by the model
    """
    boxes = []
    
    layer_names = detections.getAllLayerNames()
    
    for layer_name in layer_names:
        try:
            layer_data = detections.getLayerFp16(layer_name)
        except:
            try:
                layer_data = detections.getLayerInt32(layer_name)
            except:
                continue
        
        if layer_data is None:
            continue
        
        layer_data = np.array(layer_data)
        layer_shape = detections.getLayerDims(layer_name)
        
        num_classes = len(CHESS_CLASSES)
        expected_dims = 4 + 1 + num_classes
        
        if len(layer_shape) == 3:
            batch, num_det, dims = layer_shape
            layer_data = layer_data.reshape(batch, num_det, dims)
        elif len(layer_shape) == 2:
            num_det, dims = layer_shape
            layer_data = layer_data.reshape(num_det, dims)
        
        if layer_data.shape[-1] < expected_dims:
            continue
        
        for detection in layer_data.reshape(-1, layer_data.shape[-1]):
            if detection.shape[0] < expected_dims:
                continue
            
            x_center, y_center, width, height = detection[:4]
            objectness = detection[4] if len(detection) > 4 else 1.0
            class_scores = detection[5:5+num_classes] if len(detection) > 5 else detection[5:]
            
            if len(class_scores) < num_classes:
                continue
            
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id] * objectness
            
            if class_conf < 0.25:
                continue
            
            h, w = frame_shape[:2]
            scale_x = w / input_size
            scale_y = h / input_size
            
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(class_conf),
                    "class_id": int(class_id),
                    "class_name": CHESS_CLASSES[class_id]
                })
    
    return boxes


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input frame (BGR format)
        detections: List of detection dictionaries
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        class_id = det["class_id"]
        
        color = COLORS[class_id % len(COLORS)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1, label_size[1] + 10)
        
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0], label_y),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return frame


def run_chess_detection(blob_path: str, input_size: int = 640, conf_threshold: float = 0.25):
    """
    Run chess piece detection on OAK-D device.
    
    Args:
        blob_path: Path to the .blob model file
        input_size: Input image size for the model
        conf_threshold: Confidence threshold for detections
    """
    blob_path = Path(blob_path)
    if not blob_path.exists():
        raise FileNotFoundError(f"Blob file not found: {blob_path}")
    
    print(f"Loading model from: {blob_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Confidence threshold: {conf_threshold}")
    print("\nPress 'q' to quit")
    print("-" * 50)
    
    pipeline, cam_rgb, nn = create_chess_detection_pipeline(str(blob_path), input_size)
    
    try:
        with dai.Device() as device:
            device.startPipeline(pipeline)
            print("OAK-D device connected successfully")
            print("Using DepthAI v3 API (automatic output bridging)")
            
            rgb_output = cam_rgb.getOutputs()[0]
            rgb_queue = rgb_output.createOutputQueue(maxSize=4, blocking=False)
            nn_output = nn.getOutputs()[0]
            nn_queue = nn_output.createOutputQueue(maxSize=4, blocking=False)
            
            frame = None
            
            while True:
                nn_data = nn_queue.tryGet()
                rgb_data = rgb_queue.tryGet()
                
                if rgb_data is not None:
                    frame = rgb_data.getCvFrame()
                
                if frame is not None and nn_data is not None:
                    detections = decode_yolo_output(nn_data, frame.shape, input_size)
                    
                    filtered_detections = [
                        d for d in detections if d["confidence"] >= conf_threshold
                    ]
                    
                    annotated_frame = draw_detections(frame.copy(), filtered_detections)
                    
                    cv2.imshow("Chess Detection", annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Quitting...")
                        break
    
    except RuntimeError as e:
        if "X_LINK" in str(e):
            print(f"Error connecting to OAK-D device: {e}")
            print("\nTroubleshooting:")
            print("1. Check USB cable connection (use USB 3.0)")
            print("2. Try plugging directly into Mac port (avoid hubs)")
            print("3. Ensure depthai library is installed: pip install depthai")
            print("4. Check device permissions (see camera/permissions.sh)")
        else:
            raise
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run chess piece detection on OAK-D device"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/checkpoints/oak/best_20260515_yolo_chess_pieces_2.blob",
        help="Path to .blob model file",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    
    args = parser.parse_args()
    
    run_chess_detection(
        blob_path=args.model,
        input_size=args.input_size,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()

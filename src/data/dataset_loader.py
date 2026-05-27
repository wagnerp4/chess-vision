"""
Dataset loading utilities for chess piece detection.
Supports multiple annotation formats and converts to YOLO/COCO formats.
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import pandas as pd


class ChessDatasetLoader:
    """Loader for chess piece detection datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.annotations_dir = self.dataset_path / "annotations"
        
    def detect_annotation_format(self) -> str:
        """Detect the annotation format (YOLO, COCO, PascalVOC, or custom)."""
        if not self.annotations_dir.exists():
            return "unknown"
        
        annotation_files = list(self.annotations_dir.glob("*"))
        if not annotation_files:
            return "unknown"
        
        sample_file = annotation_files[0]
        
        if sample_file.suffix == ".txt":
            with open(sample_file, "r") as f:
                content = f.read().strip()
                if content and len(content.split()) == 5:
                    return "yolo"
        
        if sample_file.suffix == ".xml":
            return "pascal_voc"
        
        if sample_file.suffix == ".json":
            with open(sample_file, "r") as f:
                data = json.load(f)
                if "images" in data and "annotations" in data:
                    return "coco"
                if "categories" in data:
                    return "coco"
        
        return "unknown"
    
    def load_pascal_voc_annotation(self, xml_path: Path) -> List[Dict]:
        """Load Pascal VOC format annotation."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            annotations.append({
                "class": name,
                "bbox": [xmin, ymin, xmax, ymax]
            })
        
        return annotations
    
    def load_yolo_annotation(self, txt_path: Path, class_mapping: Dict[str, int]) -> List[Dict]:
        """Load YOLO format annotation."""
        annotations = []
        
        if not txt_path.exists():
            return annotations
        
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    class_name = [k for k, v in class_mapping.items() if v == class_id]
                    class_name = class_name[0] if class_name else f"class_{class_id}"
                    
                    annotations.append({
                        "class": class_name,
                        "bbox": [x_center, y_center, width, height],
                        "format": "yolo"
                    })
        
        return annotations
    
    def get_class_mapping(self) -> Dict[str, int]:
        """Get class name to ID mapping."""
        annotation_files = list(self.annotations_dir.glob("*.txt"))
        
        if not annotation_files:
            classes = ["pawn", "rook", "knight", "bishop", "queen", "king"]
            return {cls: idx for idx, cls in enumerate(classes)}
        
        all_classes = set()
        for txt_file in annotation_files:
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        all_classes.add(class_id)
        
        max_class_id = max(all_classes) if all_classes else 5
        
        classes = ["pawn", "rook", "knight", "bishop", "queen", "king"]
        if max_class_id < len(classes):
            return {cls: idx for idx, cls in enumerate(classes)}
        
        return {f"class_{i}": i for i in range(max_class_id + 1)}
    
    def convert_to_yolo_format(
        self, 
        output_dir: Path, 
        class_mapping: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """Convert annotations to YOLO format."""
        if class_mapping is None:
            class_mapping = self.get_class_mapping()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotation_format = self.detect_annotation_format()
        
        image_files = list(self.images_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            annotation_file = None
            if annotation_format == "pascal_voc":
                xml_file = self.annotations_dir / f"{img_file.stem}.xml"
                if xml_file.exists():
                    annotations = self.load_pascal_voc_annotation(xml_file)
                    
                    yolo_file = output_dir / f"{img_file.stem}.txt"
                    with open(yolo_file, "w") as f:
                        for ann in annotations:
                            class_name = ann["class"].lower()
                            if class_name not in class_mapping:
                                continue
                            
                            class_id = class_mapping[class_name]
                            xmin, ymin, xmax, ymax = ann["bbox"]
                            
                            x_center = (xmin + xmax) / 2.0 / img_width
                            y_center = (ymin + ymax) / 2.0 / img_height
                            width = (xmax - xmin) / img_width
                            height = (ymax - ymin) / img_height
                            
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            elif annotation_format == "yolo":
                txt_file = self.annotations_dir / f"{img_file.stem}.txt"
                if txt_file.exists():
                    yolo_file = output_dir / f"{img_file.stem}.txt"
                    import shutil
                    shutil.copy(txt_file, yolo_file)
        
        return class_mapping
    
    def create_dataset_yaml(self, output_path: Path, class_mapping: Dict[str, int], train_dir: str, val_dir: str, test_dir: Optional[str] = None):
        """Create YOLO dataset YAML file."""
        class_names = sorted(class_mapping.keys(), key=lambda x: class_mapping[x])
        
        yaml_content = f"""path: {output_path.parent}
train: {train_dir}
val: {val_dir}
"""
        if test_dir:
            yaml_content += f"test: {test_dir}\n"
        
        yaml_content += "\nnames:\n"
        for class_name in class_names:
            yaml_content += f"  {class_mapping[class_name]}: {class_name}\n"
        
        with open(output_path, "w") as f:
            f.write(yaml_content)
    
    def split_dataset(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.05
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split dataset into train/val/test sets."""
        image_files = list(self.images_dir.glob("*"))
        image_files = [f.stem for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        return train_files, val_files, test_files


"""
Data preprocessing utilities for chess piece detection.
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, List
import yaml


class DataPreprocessor:
    """Preprocessor for organizing dataset into train/val/test splits."""
    
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        
    def organize_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05
    ) -> str:
        """Organize dataset into YOLO format with train/val/test splits."""
        from src.data.dataset_loader import ChessDatasetLoader
        
        loader = ChessDatasetLoader(self.raw_data_path)
        
        train_files, val_files, test_files = loader.split_dataset(
            train_ratio, val_ratio, test_ratio
        )
        
        class_mapping = loader.get_class_mapping()
        
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_images_dir = self.output_path / split_name / "images"
            split_labels_dir = self.output_path / split_name / "labels"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            for file_stem in files:
                img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
                img_file = None
                for ext in img_extensions:
                    potential_file = loader.images_dir / f"{file_stem}{ext}"
                    if potential_file.exists():
                        img_file = potential_file
                        break
                
                if img_file and img_file.exists():
                    shutil.copy(img_file, split_images_dir / img_file.name)
                
                label_file = loader.annotations_dir / f"{file_stem}.txt"
                if label_file.exists():
                    shutil.copy(label_file, split_labels_dir / label_file.name)
        
        yolo_yaml_path = self.output_path / "dataset.yaml"
        loader.create_dataset_yaml(
            yolo_yaml_path,
            class_mapping,
            "train/images",
            "val/images",
            "test/images" if test_files else None
        )
        
        return str(yolo_yaml_path)


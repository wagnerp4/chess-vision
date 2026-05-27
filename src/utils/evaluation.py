from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    if not predictions or not ground_truths:
        return 0.0
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    predictions_sorted = sorted(
        predictions,
        key=lambda x: x.get("confidence", 0.0),
        reverse=True
    )
    
    gt_matched = [False] * len(ground_truths)
    
    for i, pred in enumerate(predictions_sorted):
        best_iou = 0.0
        best_gt_idx = -1
        
        pred_class = pred.get("class", "")
        pred_bbox = pred.get("bbox", [])
        
        for j, gt in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            
            gt_class = gt.get("class", "")
            gt_bbox = gt.get("bbox", [])
            
            if pred_class == gt_class:
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / max(len(ground_truths), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precisions[recalls >= r]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall) / 11.0
    
    return ap


def evaluate_model_predictions(
    predictions: Dict[str, List[Dict]],
    ground_truths: Dict[str, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    aps = []
    for image_id in ground_truths.keys():
        if image_id in predictions:
            ap = calculate_ap(
                predictions[image_id],
                ground_truths[image_id],
                iou_threshold
            )
            aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    
    return {
        "mAP": map_score,
        "AP_per_image": aps,
        "num_images": len(aps)
    }


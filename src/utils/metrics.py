"""
Segmentation metrics: IoU, Dice, Precision, Recall.
"""

from typing import Dict, Tuple

import numpy as np
import torch


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        pred: Predicted probabilities or logits
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    
    return dice.item()


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute precision and recall.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        Tuple of (precision, recall)
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision.item(), recall.item()


def compute_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute F1 score.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        F1 score
    """
    precision, recall = compute_precision_recall(pred, target, threshold, smooth)
    
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    return f1


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Dictionary of metrics
    """
    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    precision, recall = compute_precision_recall(pred, target, threshold)
    f1 = compute_f1(pred, target, threshold)
    
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Compute confusion matrix components.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with tp, fp, fn, tn counts
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = int((pred_flat * target_flat).sum().item())
    fp = int((pred_flat * (1 - target_flat)).sum().item())
    fn = int(((1 - pred_flat) * target_flat).sum().item())
    tn = int(((1 - pred_flat) * (1 - target_flat)).sum().item())
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss": [],
        }
        self.confusion = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss: float = None,
    ):
        """Update metrics with new batch."""
        batch_metrics = compute_metrics(pred, target)
        
        for key in ["iou", "dice", "precision", "recall", "f1"]:
            self.metrics[key].append(batch_metrics[key])
        
        if loss is not None:
            self.metrics["loss"].append(loss)
        
        confusion = compute_confusion_matrix(pred, target)
        for key in ["tp", "fp", "fn", "tn"]:
            self.confusion[key] += confusion[key]
    
    def get_average(self) -> Dict[str, float]:
        """Get average metrics."""
        avg = {}
        for key, values in self.metrics.items():
            if values:
                avg[key] = np.mean(values)
        return avg
    
    def get_global_metrics(self) -> Dict[str, float]:
        """Get global metrics from accumulated confusion matrix."""
        tp = self.confusion["tp"]
        fp = self.confusion["fp"]
        fn = self.confusion["fn"]
        tn = self.confusion["tn"]
        
        eps = 1e-6
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        return {
            "global_iou": iou,
            "global_precision": precision,
            "global_recall": recall,
            "global_f1": f1,
        }

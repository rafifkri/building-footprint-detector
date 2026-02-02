"""
Visualization utilities for segmentation results.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_image(
    image: Union[np.ndarray, torch.Tensor],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Args:
        image: Normalized image (C, H, W) or (H, W, C)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image as uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    mean = np.array(mean)
    std = np.array(std)
    
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create overlay of mask on image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        color: Overlay color (R, G, B)
        alpha: Overlay transparency
        
    Returns:
        Image with mask overlay
    """
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    overlay = image.copy()
    
    mask_binary = (mask > 0.5).astype(bool)
    
    if mask_binary.ndim == 3:
        mask_binary = mask_binary.squeeze()
    
    overlay[mask_binary] = (
        (1 - alpha) * overlay[mask_binary] + alpha * np.array(color)
    ).astype(np.uint8)
    
    return overlay


def create_comparison_image(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create comparison image with GT and prediction overlays.
    
    Args:
        image: RGB image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        alpha: Overlay transparency
        
    Returns:
        Comparison image with side-by-side overlays
    """
    gt_overlay = create_mask_overlay(
        image, gt_mask, color=(0, 255, 0), alpha=alpha
    )
    
    pred_overlay = create_mask_overlay(
        image, pred_mask, color=(255, 0, 0), alpha=alpha
    )
    
    comparison = np.hstack([gt_overlay, pred_overlay])
    
    return comparison


def plot_prediction(
    image: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Plot image, ground truth, and prediction side by side.
    
    Args:
        image: Input image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size
    """
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    overlay = create_mask_overlay(image, pred_mask, color=(255, 0, 0))
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_prediction_overlay(
    image: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    save_path: str,
) -> None:
    """
    Save prediction overlay image.
    
    Args:
        image: Input image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        save_path: Path to save image
    """
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()
    
    comparison = create_comparison_image(image, gt_mask, pred_mask)
    
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, comparison_bgr)


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if "train_loss" in history and "val_loss" in history:
        axes[0, 0].plot(history["train_loss"], label="Train")
        axes[0, 0].plot(history["val_loss"], label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
    
    if "train_iou" in history and "val_iou" in history:
        axes[0, 1].plot(history["train_iou"], label="Train")
        axes[0, 1].plot(history["val_iou"], label="Val")
        axes[0, 1].set_title("IoU")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
    
    if "train_dice" in history and "val_dice" in history:
        axes[1, 0].plot(history["train_dice"], label="Train")
        axes[1, 0].plot(history["val_dice"], label="Val")
        axes[1, 0].set_title("Dice")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
    
    if "lr" in history:
        axes[1, 1].plot(history["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()

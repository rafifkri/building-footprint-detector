"""
Evaluation script for building footprint segmentation.
"""

import argparse
from pathlib import Path
from typing import Dict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet_smp import load_model_from_checkpoint
from src.models.losses import get_loss_function
from src.training.dataset import BuildingDataset
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.utils.vis import save_prediction_overlay


def evaluate(
    config_path: str,
    checkpoint_path: str,
    split: str = "test",
    save_predictions: bool = False,
    output_dir: str = "output/predictions",
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        split: Data split to evaluate on
        save_predictions: Whether to save prediction visualizations
        output_dir: Directory to save predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    config = load_config(config_path)
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_config = config.get("model", {})
    model = load_model_from_checkpoint(checkpoint_path, model_config, device)
    model.eval()
    
    loss_config = config.get("loss", {})
    criterion = get_loss_function(loss_config)
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    data_config = config.get("data", {})
    tiles_dir = Path(data_config.get("tiles_dir", "data/processed/tiles"))
    
    dataset = BuildingDataset(
        images_dir=str(tiles_dir / split / "images"),
        masks_dir=str(tiles_dir / split / "masks"),
        transform=transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.get("training", {}).get("val_batch_size", 16),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
    )
    
    print(f"Evaluating on {len(dataset)} samples from {split} set")
    
    all_metrics = {
        "loss": [],
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            metrics = compute_metrics(probs, masks)
            confusion = compute_confusion_matrix(preds, masks)
            
            all_metrics["loss"].append(loss.item())
            all_metrics["iou"].append(metrics["iou"])
            all_metrics["dice"].append(metrics["dice"])
            all_metrics["precision"].append(metrics["precision"])
            all_metrics["recall"].append(metrics["recall"])
            all_metrics["f1"].append(metrics["f1"])
            
            total_tp += confusion["tp"]
            total_fp += confusion["fp"]
            total_fn += confusion["fn"]
            total_tn += confusion["tn"]
            
            if save_predictions and batch_idx < 10:
                for i in range(min(4, images.shape[0])):
                    img_idx = batch_idx * loader.batch_size + i
                    save_prediction_overlay(
                        images[i].cpu(),
                        masks[i].cpu(),
                        preds[i].cpu(),
                        str(output_path / f"pred_{img_idx:04d}.png"),
                    )
    
    final_metrics = {
        "loss": np.mean(all_metrics["loss"]),
        "iou": np.mean(all_metrics["iou"]),
        "dice": np.mean(all_metrics["dice"]),
        "precision": np.mean(all_metrics["precision"]),
        "recall": np.mean(all_metrics["recall"]),
        "f1": np.mean(all_metrics["f1"]),
    }
    
    eps = 1e-7
    global_precision = total_tp / (total_tp + total_fp + eps)
    global_recall = total_tp / (total_tp + total_fn + eps)
    global_iou = total_tp / (total_tp + total_fp + total_fn + eps)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + eps)
    
    final_metrics["global_precision"] = global_precision
    final_metrics["global_recall"] = global_recall
    final_metrics["global_iou"] = global_iou
    final_metrics["global_f1"] = global_f1
    
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Loss:              {final_metrics['loss']:.4f}")
    print(f"IoU (mean):        {final_metrics['iou']:.4f}")
    print(f"IoU (global):      {final_metrics['global_iou']:.4f}")
    print(f"Dice (mean):       {final_metrics['dice']:.4f}")
    print(f"Precision (mean):  {final_metrics['precision']:.4f}")
    print(f"Precision (global):{final_metrics['global_precision']:.4f}")
    print(f"Recall (mean):     {final_metrics['recall']:.4f}")
    print(f"Recall (global):   {final_metrics['global_recall']:.4f}")
    print(f"F1 (mean):         {final_metrics['f1']:.4f}")
    print(f"F1 (global):       {final_metrics['global_f1']:.4f}")
    print("-" * 40)
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save prediction visualizations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/predictions",
        help="Directory to save predictions",
    )
    
    args = parser.parse_args()
    
    evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.save_predictions,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

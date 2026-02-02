"""
Training script for building footprint segmentation.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.unet_smp import create_model
from src.models.losses import get_loss_function
from src.utils.config import load_config
from src.utils.metrics import compute_metrics
from src.training.dataset import BuildingDataset


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(config: dict, split: str) -> A.Compose:
    """Get augmentation transforms based on config."""
    if split == "train":
        aug_config = config.get("augmentation", {}).get("train", {})
        transforms_list = [
            A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug_config.get("vertical_flip", 0.5)),
            A.RandomRotate90(p=aug_config.get("rotate90", 0.5)),
        ]
        
        if "shift_scale_rotate" in aug_config:
            ssr = aug_config["shift_scale_rotate"]
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=ssr.get("shift_limit", 0.1),
                    scale_limit=ssr.get("scale_limit", 0.1),
                    rotate_limit=ssr.get("rotate_limit", 15),
                    p=ssr.get("p", 0.5),
                )
            )
        
        if "random_brightness_contrast" in aug_config:
            rbc = aug_config["random_brightness_contrast"]
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=rbc.get("brightness_limit", 0.2),
                    contrast_limit=rbc.get("contrast_limit", 0.2),
                    p=rbc.get("p", 0.3),
                )
            )
        
        if "gaussian_noise" in aug_config:
            gn = aug_config["gaussian_noise"]
            transforms_list.append(
                A.GaussNoise(
                    var_limit=tuple(gn.get("var_limit", [10.0, 50.0])),
                    p=gn.get("p", 0.2),
                )
            )
        
        transforms_list.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        transforms_list = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms_list)


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Get optimizer based on config."""
    opt_config = config.get("optimizer", {})
    opt_name = opt_config.get("name", "adamw").lower()
    lr = opt_config.get("lr", 0.0001)
    weight_decay = opt_config.get("weight_decay", 0.01)
    
    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = opt_config.get("momentum", 0.9)
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Get learning rate scheduler based on config."""
    sched_config = config.get("scheduler", {})
    sched_name = sched_config.get("name", "cosine").lower()
    
    if sched_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get("T_max", 100),
            eta_min=sched_config.get("eta_min", 1e-6),
        )
    elif sched_name == "step":
        return StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 30),
            gamma=sched_config.get("gamma", 0.1),
        )
    elif sched_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get("mode", "max"),
            factor=sched_config.get("factor", 0.1),
            patience=sched_config.get("patience", 10),
        )
    else:
        return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            metrics = compute_metrics(preds, masks)
        
        total_loss += loss.item() * accumulation_steps
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "iou": f"{total_iou / num_batches:.4f}",
        })
    
    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        preds = torch.sigmoid(outputs)
        metrics = compute_metrics(preds, masks)
        
        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "iou": f"{total_iou / num_batches:.4f}",
        })
    
    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    config: dict,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)


def train(config_path: str) -> None:
    """Main training function."""
    config = load_config(config_path)
    
    seed = config.get("seed", 42)
    set_seed(seed)
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_config = config.get("model", {})
    model = create_model(
        name=model_config.get("name", "unet"),
        encoder=model_config.get("encoder", "resnet34"),
        encoder_weights=model_config.get("encoder_weights", "imagenet"),
        in_channels=model_config.get("in_channels", 3),
        classes=model_config.get("classes", 1),
        activation=model_config.get("activation", None),
    )
    model.to(device)
    
    loss_config = config.get("loss", {})
    criterion = get_loss_function(loss_config)
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    train_config = config.get("training", {})
    mixed_precision = train_config.get("mixed_precision", True)
    scaler = GradScaler() if mixed_precision and device == "cuda" else None
    
    data_config = config.get("data", {})
    tiles_dir = Path(data_config.get("tiles_dir", "data/processed/tiles"))
    
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")
    
    train_dataset = BuildingDataset(
        images_dir=str(tiles_dir / "train" / "images"),
        masks_dir=str(tiles_dir / "train" / "masks"),
        transform=train_transform,
    )
    
    val_dataset = BuildingDataset(
        images_dir=str(tiles_dir / "val" / "images"),
        masks_dir=str(tiles_dir / "val" / "masks"),
        transform=val_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("val_batch_size", 16),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    checkpoint_config = config.get("checkpoint", {})
    checkpoint_dir = Path(checkpoint_config.get("dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logging_config = config.get("logging", {})
    writer = None
    if logging_config.get("tensorboard", True):
        log_dir = logging_config.get("tensorboard_dir", "logs")
        writer = SummaryWriter(log_dir)
    
    epochs = train_config.get("epochs", 100)
    early_stopping_patience = train_config.get("early_stopping_patience", 15)
    accumulation_steps = train_config.get("accumulation_steps", 1)
    grad_clip = train_config.get("grad_clip", 1.0)
    
    monitor = checkpoint_config.get("monitor", "val_iou")
    mode = checkpoint_config.get("mode", "max")
    
    best_metric = float("-inf") if mode == "max" else float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
        )
        
        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.get(monitor.replace("val_", ""), val_metrics["iou"]))
            else:
                scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")
        
        if writer:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/iou", train_metrics["iou"], epoch)
            writer.add_scalar("train/dice", train_metrics["dice"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/iou", val_metrics["iou"], epoch)
            writer.add_scalar("val/dice", val_metrics["dice"], epoch)
            writer.add_scalar("val/precision", val_metrics["precision"], epoch)
            writer.add_scalar("val/recall", val_metrics["recall"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        current_metric = val_metrics.get(monitor.replace("val_", ""), val_metrics["iou"])
        
        is_best = (mode == "max" and current_metric > best_metric) or \
                  (mode == "min" and current_metric < best_metric)
        
        if is_best:
            best_metric = current_metric
            patience_counter = 0
            
            if checkpoint_config.get("save_best", True):
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    str(checkpoint_dir / "best.pth"), config,
                )
                print(f"Saved best model with {monitor}: {best_metric:.4f}")
        else:
            patience_counter += 1
        
        if checkpoint_config.get("save_last", True):
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                str(checkpoint_dir / "last.pth"), config,
            )
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    if writer:
        writer.close()
    
    print(f"\nTraining complete. Best {monitor}: {best_metric:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to config file",
    )
    
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

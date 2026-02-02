"""
Loss functions for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance."""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss."""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return (1.0 - tversky) ** self.gamma


class LovaszHingeLoss(nn.Module):
    """Lovasz hinge loss for binary segmentation."""
    
    def __init__(self):
        super().__init__()
    
    def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        signs = 2.0 * target - 1.0
        errors = 1.0 - pred * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = target[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        
        return loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Get loss function based on configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss function module
    """
    loss_name = config.get("name", "dice_bce").lower()
    
    if loss_name == "dice":
        return DiceLoss(smooth=config.get("smooth", 1.0))
    
    elif loss_name in ["dice_bce", "bce_dice"]:
        return BCEDiceLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            smooth=config.get("smooth", 1.0),
        )
    
    elif loss_name == "focal":
        return FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
        )
    
    elif loss_name == "tversky":
        return TverskyLoss(
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.5),
            smooth=config.get("smooth", 1.0),
        )
    
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.5),
            gamma=config.get("gamma", 1.0),
            smooth=config.get("smooth", 1.0),
        )
    
    elif loss_name == "lovasz":
        return LovaszHingeLoss()
    
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

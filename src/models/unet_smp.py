"""
U-Net model factory using segmentation_models_pytorch.
"""

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_model(
    name: str = "unet",
    encoder: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[str] = None,
) -> nn.Module:
    """
    Create a segmentation model using segmentation_models_pytorch.
    
    Args:
        name: Model architecture ('unet', 'unetplusplus', 'deeplabv3', 'fpn', 'pan', 'linknet')
        encoder: Encoder backbone name
        encoder_weights: Pretrained weights for encoder
        in_channels: Number of input channels
        classes: Number of output classes
        activation: Output activation function
        
    Returns:
        PyTorch model
    """
    model_factory = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pan": smp.PAN,
        "linknet": smp.Linknet,
        "pspnet": smp.PSPNet,
        "manet": smp.MAnet,
    }
    
    if name.lower() not in model_factory:
        raise ValueError(f"Unknown model: {name}. Available: {list(model_factory.keys())}")
    
    model_class = model_factory[name.lower()]
    
    model = model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: dict,
    device: str = "cuda",
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration dictionary
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    model = create_model(
        name=model_config.get("name", "unet"),
        encoder=model_config.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=model_config.get("in_channels", 3),
        classes=model_config.get("classes", 1),
        activation=model_config.get("activation", None),
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


class SegmentationModel(nn.Module):
    """Wrapper class for segmentation models with additional functionality."""
    
    def __init__(
        self,
        name: str = "unet",
        encoder: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.model = create_model(
            name=name,
            encoder=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        self.name = name
        self.encoder = encoder
        self.in_channels = in_channels
        self.classes = classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predict with thresholding."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        return preds
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "name": self.name,
            "encoder": self.encoder,
            "in_channels": self.in_channels,
            "classes": self.classes,
        }

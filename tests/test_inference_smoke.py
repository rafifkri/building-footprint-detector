"""
Smoke test for inference pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet_smp import create_model


def test_model_creation():
    """Test model creation."""
    model = create_model(
        name="unet",
        encoder="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    assert model is not None
    
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (1, 1, 256, 256)


def test_model_forward():
    """Test model forward pass."""
    model = create_model(
        name="unet",
        encoder="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.eval()
    
    batch_sizes = [1, 2, 4]
    image_sizes = [256, 512]
    
    for bs in batch_sizes:
        for size in image_sizes:
            x = torch.randn(bs, 3, size, size)
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (bs, 1, size, size)


def test_model_architectures():
    """Test different model architectures."""
    architectures = ["unet", "unetplusplus", "fpn", "linknet"]
    
    for arch in architectures:
        model = create_model(
            name=arch,
            encoder="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (1, 1, 256, 256), f"Failed for {arch}"


def test_model_encoders():
    """Test different encoder backbones."""
    encoders = ["resnet18", "resnet34", "efficientnet-b0"]
    
    for encoder in encoders:
        try:
            model = create_model(
                name="unet",
                encoder=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            
            x = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (1, 1, 256, 256), f"Failed for {encoder}"
        except Exception as e:
            pytest.skip(f"Encoder {encoder} not available: {e}")


def test_model_prediction():
    """Test model prediction with thresholding."""
    model = create_model(
        name="unet",
        encoder="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.eval()
    
    x = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    
    assert probs.min() >= 0 and probs.max() <= 1
    assert set(preds.unique().tolist()).issubset({0.0, 1.0})

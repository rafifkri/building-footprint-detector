"""
Test dataset loading.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset import BuildingDataset


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = Path(tmpdir) / "images"
        masks_dir = Path(tmpdir) / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        
        for i in range(5):
            image = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
            mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
            
            transform = from_bounds(0, 0, 512, 512, 512, 512)
            
            image_path = images_dir / f"tile_{i:04d}.tif"
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=512,
                width=512,
                count=3,
                dtype="uint8",
                transform=transform,
            ) as dst:
                dst.write(image)
            
            mask_path = masks_dir / f"tile_{i:04d}_mask.tif"
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=512,
                width=512,
                count=1,
                dtype="uint8",
                transform=transform,
            ) as dst:
                dst.write(mask, 1)
        
        yield str(images_dir), str(masks_dir)


def test_dataset_init(sample_data):
    """Test dataset initialization."""
    images_dir, masks_dir = sample_data
    
    dataset = BuildingDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    
    assert len(dataset) == 5


def test_dataset_getitem(sample_data):
    """Test dataset item retrieval."""
    images_dir, masks_dir = sample_data
    
    dataset = BuildingDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    
    image, mask = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape[0] == 3
    assert mask.shape[0] == 1


def test_dataset_with_transform(sample_data):
    """Test dataset with albumentations transform."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    images_dir, masks_dir = sample_data
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    dataset = BuildingDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform,
    )
    
    image, mask = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32


def test_dataset_empty():
    """Test dataset with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = Path(tmpdir) / "images"
        masks_dir = Path(tmpdir) / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()
        
        dataset = BuildingDataset(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
        )
        
        assert len(dataset) == 0

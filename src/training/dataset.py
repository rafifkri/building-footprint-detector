"""
Dataset class for building footprint segmentation.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    """Dataset for loading building footprint images and masks."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_suffix: str = ".tif",
        mask_suffix: str = "_mask.tif",
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing image tiles
            masks_dir: Directory containing mask tiles
            transform: Albumentations transform
            image_suffix: Image file suffix
            mask_suffix: Mask file suffix
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        
        self.image_files = sorted([
            f for f in self.images_dir.glob(f"*{image_suffix}")
            if not f.name.endswith(mask_suffix)
        ])
        
        if len(self.image_files) == 0:
            self.image_files = sorted([
                f for f in self.images_dir.glob("*.tif")
            ])
            self.image_files.extend(sorted([
                f for f in self.images_dir.glob("*.png")
            ]))
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        try:
            with rasterio.open(path) as src:
                image = src.read()
                if image.shape[0] >= 3:
                    image = image[:3]
                elif image.shape[0] == 1:
                    # Convert grayscale to RGB
                    image = np.repeat(image, 3, axis=0)
                image = np.transpose(image, (1, 2, 0))
        except Exception:
            # Fallback to PIL
            from PIL import Image
            img = Image.open(path)
            image = np.array(img)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[:, :, :3]
        return image.astype(np.float32)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask from file."""
        try:
            with rasterio.open(path) as src:
                mask = src.read(1)
        except Exception:
            # Try with PIL as fallback
            from PIL import Image
            mask = np.array(Image.open(path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.float32)
        return mask
    
    def _find_mask_path(self, image_path: Path) -> Path:
        """Find corresponding mask path for an image."""
        image_name = image_path.stem
        
        if image_name.endswith("_mask"):
            mask_name = image_name
        else:
            mask_name = f"{image_name}_mask"
        
        mask_path = self.masks_dir / f"{mask_name}.tif"
        if mask_path.exists():
            return mask_path
        
        mask_path = self.masks_dir / f"{image_name}{self.mask_suffix}"
        if mask_path.exists():
            return mask_path
        
        mask_path = self.masks_dir / f"{image_name}.tif"
        if mask_path.exists():
            return mask_path
        
        mask_path = self.masks_dir / f"{image_name}.png"
        if mask_path.exists():
            return mask_path
        
        raise FileNotFoundError(f"Mask not found for {image_path}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        mask_path = self._find_mask_path(image_path)
        
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Normalize image to 0-255 range if needed for albumentations
        if image.max() <= 1.0 and image.max() > 0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            mask = torch.from_numpy(mask)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # Ensure mask is float
        if mask.dtype != torch.float32:
            mask = mask.float()
        
        return image, mask


class InferenceDataset(Dataset):
    """Dataset for inference on large images (tiled)."""
    
    def __init__(
        self,
        tiles: list,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize inference dataset.
        
        Args:
            tiles: List of (tile_data, tile_info) tuples
            transform: Albumentations transform
        """
        self.tiles = tiles
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        tile_data, tile_info = self.tiles[idx]
        
        # Ensure HWC format for albumentations
        if tile_data.ndim == 3 and tile_data.shape[0] <= 4:
            tile_data = np.transpose(tile_data, (1, 2, 0))
        
        image = tile_data.astype(np.float32)
        
        # Normalize to 0-255 if needed
        if image.max() <= 1.0 and image.max() > 0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        
        return image, tile_info

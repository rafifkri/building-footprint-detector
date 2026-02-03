"""
Inference script for large images with tiling and TTA.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.models.unet_smp import load_model_from_checkpoint
from src.utils.config import load_config


class TiledInference:
    """Handle inference on large images with tiling."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        tile_size: int = 512,
        overlap: int = 128,
        batch_size: int = 8,
        threshold: float = 0.5,
        use_tta: bool = True,
    ):
        """
        Initialize tiled inference.
        
        Args:
            model: Trained segmentation model
            device: Device to run inference on
            tile_size: Size of each tile
            overlap: Overlap between tiles
            batch_size: Batch size for inference
            threshold: Threshold for binary mask
            use_tta: Whether to use test-time augmentation
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.threshold = threshold
        self.use_tta = use_tta
        
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.model.eval()
    
    def _calculate_tiles(
        self,
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions."""
        tiles = []
        stride = self.tile_size - self.overlap
        
        for row_off in range(0, height, stride):
            for col_off in range(0, width, stride):
                row_end = min(row_off + self.tile_size, height)
                col_end = min(col_off + self.tile_size, width)
                
                if row_end - row_off < self.tile_size:
                    row_off = max(0, row_end - self.tile_size)
                if col_end - col_off < self.tile_size:
                    col_off = max(0, col_end - self.tile_size)
                
                tiles.append((col_off, row_off, col_end - col_off, row_end - row_off))
        
        return tiles
    
    def _extract_tile(
        self,
        image: np.ndarray,
        col_off: int,
        row_off: int,
        tile_w: int,
        tile_h: int,
    ) -> np.ndarray:
        """Extract a tile from the image."""
        tile = image[row_off:row_off + tile_h, col_off:col_off + tile_w]
        
        if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        
        return tile
    
    def _apply_tta(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply test-time augmentation."""
        predictions = []
        
        with torch.no_grad():
            # Original
            pred = torch.sigmoid(self.model(batch))
            predictions.append(pred)
            
            # Horizontal flip
            flipped_h = torch.flip(batch, dims=[3])
            pred_h = torch.sigmoid(self.model(flipped_h))
            pred_h = torch.flip(pred_h, dims=[3])
            predictions.append(pred_h)
            
            # Vertical flip
            flipped_v = torch.flip(batch, dims=[2])
            pred_v = torch.sigmoid(self.model(flipped_v))
            pred_v = torch.flip(pred_v, dims=[2])
            predictions.append(pred_v)
            
            # 90 degree rotation
            rotated = torch.rot90(batch, k=1, dims=[2, 3])
            pred_r = torch.sigmoid(self.model(rotated))
            pred_r = torch.rot90(pred_r, k=-1, dims=[2, 3])
            predictions.append(pred_r)
        
        avg_pred = torch.stack(predictions, dim=0).mean(dim=0)
        return avg_pred
    
    def predict(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Run inference on a large image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output mask (optional)
            
        Returns:
            Predicted mask and metadata
        """
        with rasterio.open(image_path) as src:
            image = src.read()
            meta = src.meta.copy()
            profile = src.profile.copy()
        
        if image.shape[0] >= 3:
            image = image[:3]
        image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        
        height, width = image.shape[:2]
        
        tiles = self._calculate_tiles(width, height)
        
        pred_sum = np.zeros((height, width), dtype=np.float32)
        pred_count = np.zeros((height, width), dtype=np.float32)
        
        tile_batches = []
        tile_positions = []
        
        for col_off, row_off, tile_w, tile_h in tiles:
            tile = self._extract_tile(image, col_off, row_off, tile_w, tile_h)
            transformed = self.transform(image=tile)
            tile_tensor = transformed["image"]
            
            tile_batches.append(tile_tensor)
            tile_positions.append((col_off, row_off, tile_w, tile_h))
            
            if len(tile_batches) >= self.batch_size:
                self._process_batch(
                    tile_batches,
                    tile_positions,
                    pred_sum,
                    pred_count,
                )
                tile_batches = []
                tile_positions = []
        
        if tile_batches:
            self._process_batch(
                tile_batches,
                tile_positions,
                pred_sum,
                pred_count,
            )
        
        pred_count = np.maximum(pred_count, 1)
        pred_mask = pred_sum / pred_count
        
        binary_mask = (pred_mask > self.threshold).astype(np.uint8) * 255
        
        if output_path:
            output_meta = meta.copy()
            output_meta.update({
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw",
            })
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(output_path, "w", **output_meta) as dst:
                dst.write(binary_mask, 1)
        
        return binary_mask, {
            "width": width,
            "height": height,
            "crs": str(meta.get("crs")),
            "transform": meta.get("transform"),
        }
    
    def _process_batch(
        self,
        tile_batches: List[torch.Tensor],
        tile_positions: List[Tuple[int, int, int, int]],
        pred_sum: np.ndarray,
        pred_count: np.ndarray,
    ) -> None:
        """Process a batch of tiles."""
        batch = torch.stack(tile_batches).to(self.device)
        
        with torch.no_grad():
            if self.use_tta:
                preds = self._apply_tta(batch)
            else:
                preds = torch.sigmoid(self.model(batch))
        
        preds = preds.cpu().numpy()
        
        for pred, (col_off, row_off, tile_w, tile_h) in zip(preds, tile_positions):
            pred = pred.squeeze()
            pred = pred[:tile_h, :tile_w]
            
            pred_sum[row_off:row_off + tile_h, col_off:col_off + tile_w] += pred
            pred_count[row_off:row_off + tile_h, col_off:col_off + tile_w] += 1


def run_inference(
    config_path: str,
    input_path: str,
    output_path: str,
) -> None:
    """
    Run inference on an image.
    
    Args:
        config_path: Path to inference config
        input_path: Path to input image
        output_path: Path to output mask
    """
    config = load_config(config_path)
    
    device = config.get("inference", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    
    model_config = config.get("model", {})
    checkpoint_path = config.get("checkpoint", {}).get("path", "checkpoints/best.pth")
    model = load_model_from_checkpoint(checkpoint_path, model_config, device)
    
    infer_config = config.get("inference", {})
    tta_config = config.get("tta", {})
    
    inferencer = TiledInference(
        model=model,
        device=device,
        tile_size=infer_config.get("tile_size", 512),
        overlap=infer_config.get("overlap", 128),
        batch_size=infer_config.get("batch_size", 8),
        threshold=infer_config.get("threshold", 0.5),
        use_tta=tta_config.get("enabled", True),
    )
    
    print(f"Processing: {input_path}")
    mask, metadata = inferencer.predict(input_path, output_path)
    print(f"Output saved to: {output_path}")
    print(f"Image size: {metadata['width']}x{metadata['height']}")
    
    return mask, metadata


def main():
    parser = argparse.ArgumentParser(description="Run inference on large images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer.yaml",
        help="Path to inference config",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output mask",
    )
    
    args = parser.parse_args()
    run_inference(args.config, args.input, args.output)


if __name__ == "__main__":
    main()

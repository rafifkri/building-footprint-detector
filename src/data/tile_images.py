"""
Tile large images and masks into smaller patches for training.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def calculate_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list:
    """
    Calculate tile windows for an image.
    
    Args:
        width: Image width
        height: Image height
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        List of (col_off, row_off, tile_width, tile_height) tuples
    """
    tiles = []
    stride = tile_size - overlap
    
    for row_off in range(0, height, stride):
        for col_off in range(0, width, stride):
            tile_width = min(tile_size, width - col_off)
            tile_height = min(tile_size, height - row_off)
            
            if tile_width < tile_size // 2 or tile_height < tile_size // 2:
                continue
                
            tiles.append((col_off, row_off, tile_width, tile_height))
    
    return tiles


def pad_tile(
    tile: np.ndarray,
    target_size: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Pad tile to target size."""
    if tile.ndim == 2:
        h, w = tile.shape
        padded = np.full((target_size, target_size), pad_value, dtype=tile.dtype)
        padded[:h, :w] = tile
    else:
        c, h, w = tile.shape
        padded = np.full((c, target_size, target_size), pad_value, dtype=tile.dtype)
        padded[:, :h, :w] = tile
    return padded


def tile_image_and_mask(
    image_path: str,
    mask_path: str,
    output_image_dir: str,
    output_mask_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_building_ratio: float = 0.0,
    base_name: str = None,
) -> int:
    """
    Tile a single image and its corresponding mask.
    
    Args:
        image_path: Path to input image
        mask_path: Path to input mask
        output_image_dir: Directory to save image tiles
        output_mask_dir: Directory to save mask tiles
        tile_size: Size of each tile
        overlap: Overlap between tiles
        min_building_ratio: Minimum ratio of building pixels to include tile
        base_name: Base name for output files
        
    Returns:
        Number of tiles created
    """
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
    
    if base_name is None:
        base_name = Path(image_path).stem
    
    with rasterio.open(image_path) as src:
        image_meta = src.meta.copy()
        width = src.width
        height = src.height
    
    tiles = calculate_tiles(width, height, tile_size, overlap)
    tile_count = 0
    
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        for idx, (col_off, row_off, tile_w, tile_h) in enumerate(tiles):
            window = Window(col_off, row_off, tile_w, tile_h)
            
            image_tile = img_src.read(window=window)
            mask_tile = mask_src.read(1, window=window)
            
            if min_building_ratio > 0:
                building_ratio = np.sum(mask_tile > 0) / (tile_w * tile_h)
                if building_ratio < min_building_ratio:
                    continue
            
            if tile_w < tile_size or tile_h < tile_size:
                image_tile = pad_tile(image_tile, tile_size, 0)
                mask_tile = pad_tile(mask_tile, tile_size, 0)
            
            tile_name = f"{base_name}_tile_{idx:04d}"
            
            img_meta = image_meta.copy()
            img_meta.update({
                "width": tile_size,
                "height": tile_size,
                "transform": rasterio.windows.transform(window, img_src.transform),
            })
            
            image_output = Path(output_image_dir) / f"{tile_name}.tif"
            with rasterio.open(image_output, "w", **img_meta) as dst:
                dst.write(image_tile)
            
            mask_meta = img_meta.copy()
            mask_meta.update({
                "count": 1,
                "dtype": "uint8",
            })
            
            mask_output = Path(output_mask_dir) / f"{tile_name}_mask.tif"
            with rasterio.open(mask_output, "w", **mask_meta) as dst:
                dst.write(mask_tile[np.newaxis, :, :] if mask_tile.ndim == 2 else mask_tile)
            
            tile_count += 1
    
    return tile_count


def find_mask_for_image(image_path: str, masks_dir: str = None) -> str:
    """
    Find corresponding mask file for an image.
    Supports multiple naming conventions and extensions.
    
    Args:
        image_path: Path to the image file
        masks_dir: Optional directory to search for masks
        
    Returns:
        Path to mask file or None if not found
    """
    image_path = Path(image_path)
    image_stem = image_path.stem
    image_parent = image_path.parent
    
    possible_mask_dirs = []
    if masks_dir:
        possible_mask_dirs.append(Path(masks_dir))
    
    if image_parent.name == "images":
        possible_mask_dirs.append(image_parent.parent / "mask")
        possible_mask_dirs.append(image_parent.parent / "masks")
    
    possible_mask_dirs.append(image_parent)
    
    possible_extensions = [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
    
    for mask_dir in possible_mask_dirs:
        if not mask_dir.exists():
            continue
        
        for ext in possible_extensions:
            mask_path = mask_dir / f"{image_stem}{ext}"
            if mask_path.exists():
                return str(mask_path)
            
            mask_path = mask_dir / f"{image_stem}_mask{ext}"
            if mask_path.exists():
                return str(mask_path)
    
    return None


def tile_from_splits(
    splits_dir: str,
    masks_dir: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_building_ratio: float = 0.0,
) -> None:
    """
    Tile all images from train/val/test splits.
    
    Args:
        splits_dir: Directory containing split CSVs
        masks_dir: Directory containing rasterized masks (optional, will auto-detect)
        output_dir: Base output directory for tiles
        tile_size: Size of each tile
        overlap: Overlap between tiles
        min_building_ratio: Minimum building pixel ratio
    """
    splits_path = Path(splits_dir)
    masks_path = Path(masks_dir) if masks_dir else None
    output_path = Path(output_dir)
    
    for split in ["train", "val", "test"]:
        split_csv = splits_path / f"{split}.csv"
        if not split_csv.exists():
            print(f"Warning: {split_csv} not found, skipping")
            continue
        
        df = pd.read_csv(split_csv)
        
        split_image_dir = output_path / split / "images"
        split_mask_dir = output_path / split / "masks"
        
        total_tiles = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Tiling {split}"):
            image_path = row["image_path"]
            city = row.get("city", "unknown")
            image_name = Path(image_path).stem
            
            mask_path = find_mask_for_image(image_path, str(masks_path) if masks_path else None)
            
            if mask_path is None:
                print(f"Warning: Mask not found for {image_path}")
                continue
            
            base_name = f"{city}_{image_name}"
            
            try:
                tiles = tile_image_and_mask(
                    image_path,
                    mask_path,
                    str(split_image_dir),
                    str(split_mask_dir),
                    tile_size,
                    overlap,
                    min_building_ratio,
                    base_name,
                )
                total_tiles += tiles
            except Exception as e:
                print(f"Error tiling {image_path}: {e}")
        
        print(f"{split}: {total_tiles} tiles created")


def main():
    parser = argparse.ArgumentParser(description="Tile images and masks")
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits",
        help="Directory containing split CSVs",
    )
    parser.add_argument(
        "--masks",
        type=str,
        default="data/processed/full_masks",
        help="Directory containing rasterized masks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/tiles",
        help="Output directory for tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between tiles in pixels",
    )
    parser.add_argument(
        "--min-building-ratio",
        type=float,
        default=0.0,
        help="Minimum building pixel ratio to include tile",
    )
    
    args = parser.parse_args()
    
    tile_from_splits(
        args.splits,
        args.masks,
        args.output,
        args.tile_size,
        args.overlap,
        args.min_building_ratio,
    )


if __name__ == "__main__":
    main()

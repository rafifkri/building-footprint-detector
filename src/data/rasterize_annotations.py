"""
Rasterize vector annotations (GeoJSON/Shapefile) to binary masks.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from tqdm import tqdm


def rasterize_geojson(
    geojson_path: str,
    reference_image_path: str,
    output_path: str,
    burn_value: int = 255,
) -> None:
    """
    Rasterize GeoJSON polygons to a binary mask aligned with reference image.
    
    Args:
        geojson_path: Path to GeoJSON file with building polygons
        reference_image_path: Path to reference image for dimensions and CRS
        output_path: Path to output mask file
        burn_value: Value to burn for building pixels
    """
    with rasterio.open(reference_image_path) as src:
        meta = src.meta.copy()
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
    
    gdf = gpd.read_file(geojson_path)
    
    if gdf.crs is not None and crs is not None:
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if len(gdf) > 0:
        shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None]
        
        if shapes:
            mask = features.rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
    
    meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
    })
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask, 1)


def rasterize_from_manifest(
    manifest_path: str,
    output_dir: str,
    burn_value: int = 255,
) -> None:
    """
    Rasterize all annotations from manifest CSV.
    
    Args:
        manifest_path: Path to manifest CSV
        output_dir: Directory to save rasterized masks
        burn_value: Value to burn for building pixels
    """
    df = pd.read_csv(manifest_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rasterizing annotations"):
        image_path = row["image_path"]
        annotation_path = row.get("annotation_path", "")
        
        if pd.isna(annotation_path) or annotation_path == "":
            skip_count += 1
            continue
        
        if not Path(annotation_path).exists():
            print(f"Warning: Annotation not found: {annotation_path}")
            skip_count += 1
            continue
        
        image_name = Path(image_path).stem
        city = row.get("city", "unknown")
        mask_filename = f"{city}_{image_name}_mask.tif"
        mask_path = output_path / mask_filename
        
        try:
            rasterize_geojson(
                annotation_path,
                image_path,
                str(mask_path),
                burn_value,
            )
            success_count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            error_count += 1
    
    print(f"Rasterization complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description="Rasterize vector annotations to masks")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/full_masks",
        help="Output directory for rasterized masks",
    )
    parser.add_argument(
        "--burn-value",
        type=int,
        default=255,
        help="Value to burn for building pixels",
    )
    
    args = parser.parse_args()
    rasterize_from_manifest(args.manifest, args.output, args.burn_value)


if __name__ == "__main__":
    main()

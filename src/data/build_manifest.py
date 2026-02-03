"""
Build manifest CSV from raw data directory.
Scans raw data folder and creates image-annotation pairs.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import rasterio


def get_image_metadata(image_path: str) -> dict:
    """Extract metadata from a raster image."""
    metadata = {
        "width": None,
        "height": None,
        "crs": None,
        "transform_json": None,
    }
    
    try:
        with rasterio.open(image_path) as src:
            metadata["width"] = src.width
            metadata["height"] = src.height
            metadata["crs"] = str(src.crs) if src.crs else None
            if src.transform:
                transform_list = list(src.transform)[:6]
                metadata["transform_json"] = json.dumps(transform_list)
    except Exception as e:
        print(f"Warning: Could not read metadata from {image_path}: {e}")
    
    return metadata


def find_annotation_file(image_path: Path, annotations_dir: Path) -> str:
    """Find matching annotation file for an image."""
    image_stem = image_path.stem
    
    annotation_extensions = [".geojson", ".json", ".shp"]
    
    for ext in annotation_extensions:
        annotation_path = annotations_dir / f"{image_stem}{ext}"
        if annotation_path.exists():
            return str(annotation_path)
    
    for ext in annotation_extensions:
        for annotation_file in annotations_dir.glob(f"*{ext}"):
            if image_stem in annotation_file.stem:
                return str(annotation_file)
    
    return ""


def build_manifest(input_dir: str, output_path: str) -> pd.DataFrame:
    """
    Build manifest CSV from raw data directory.
    
    Args:
        input_dir: Path to raw data directory containing city folders
        output_path: Path to output manifest CSV
        
    Returns:
        DataFrame with manifest data
    """
    input_path = Path(input_dir)
    records = []
    
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    
    for city_dir in input_path.iterdir():
        if not city_dir.is_dir():
            continue
            
        city_name = city_dir.name
        images_dir = city_dir / "images"
        annotations_dir = city_dir / "annotations"
        
        if not images_dir.exists():
            print(f"Warning: No images directory found for {city_name}")
            continue
        
        for image_file in images_dir.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue
            
            image_path = str(image_file)
            
            annotation_path = ""
            if annotations_dir.exists():
                annotation_path = find_annotation_file(image_file, annotations_dir)
            
            metadata = get_image_metadata(image_path)
            
            record = {
                "image_path": image_path,
                "annotation_path": annotation_path,
                "city": city_name,
                "width": metadata["width"],
                "height": metadata["height"],
                "crs": metadata["crs"],
                "transform_json": metadata["transform_json"],
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Manifest saved to {output_path}")
    print(f"Total images: {len(df)}")
    
    if len(df) > 0:
        print(f"Cities: {df['city'].unique().tolist()}")
    else:
        print("Warning: No images found. Check your input directory structure.")
        print("Expected structure: {input_dir}/{city}/images/*.tif")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build manifest CSV from raw data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/manifest.csv",
        help="Path to output manifest CSV",
    )
    
    args = parser.parse_args()
    build_manifest(args.input, args.output)


if __name__ == "__main__":
    main()

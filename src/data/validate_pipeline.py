"""
Validate the data pipeline and report any issues.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import rasterio
from tqdm import tqdm


def validate_manifest(manifest_path: str) -> Dict:
    """Validate manifest CSV."""
    results = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Cannot read manifest: {e}")
        return results
    
    results["stats"]["total_rows"] = len(df)
    results["stats"]["cities"] = df["city"].unique().tolist() if "city" in df.columns else []
    
    # Check required columns
    required_cols = ["image_path", "city"]
    for col in required_cols:
        if col not in df.columns:
            results["issues"].append(f"Missing required column: {col}")
            results["valid"] = False
    
    # Check for missing annotations
    if "annotation_path" in df.columns:
        missing_annotations = df["annotation_path"].isna() | (df["annotation_path"] == "")
        results["stats"]["missing_annotations"] = missing_annotations.sum()
        
        if missing_annotations.sum() > 0:
            results["issues"].append(
                f"{missing_annotations.sum()} images have no annotation path"
            )
    
    # Verify image paths exist
    missing_images = []
    for idx, row in df.iterrows():
        if not Path(row["image_path"]).exists():
            missing_images.append(row["image_path"])
    
    results["stats"]["missing_images"] = len(missing_images)
    if missing_images:
        results["issues"].append(f"{len(missing_images)} images not found on disk")
        results["valid"] = False
    
    return results


def validate_splits(splits_dir: str, manifest_path: str) -> Dict:
    """Validate train/val/test splits."""
    results = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    splits_path = Path(splits_dir)
    
    for split in ["train", "val", "test"]:
        split_file = splits_path / f"{split}.csv"
        
        if not split_file.exists():
            results["issues"].append(f"Missing split file: {split}.csv")
            results["valid"] = False
            continue
        
        df = pd.read_csv(split_file)
        results["stats"][f"{split}_count"] = len(df)
    
    # Check for data leakage
    try:
        train_df = pd.read_csv(splits_path / "train.csv")
        val_df = pd.read_csv(splits_path / "val.csv")
        test_df = pd.read_csv(splits_path / "test.csv")
        
        train_images = set(train_df["image_path"])
        val_images = set(val_df["image_path"])
        test_images = set(test_df["image_path"])
        
        train_val_overlap = train_images & val_images
        train_test_overlap = train_images & test_images
        val_test_overlap = val_images & test_images
        
        if train_val_overlap:
            results["issues"].append(f"Data leakage: {len(train_val_overlap)} images in both train and val")
            results["valid"] = False
        
        if train_test_overlap:
            results["issues"].append(f"Data leakage: {len(train_test_overlap)} images in both train and test")
            results["valid"] = False
        
        if val_test_overlap:
            results["issues"].append(f"Data leakage: {len(val_test_overlap)} images in both val and test")
            results["valid"] = False
            
    except Exception as e:
        results["issues"].append(f"Error checking data leakage: {e}")
    
    return results


def validate_tiles(tiles_dir: str, expected_size: int = 512) -> Dict:
    """Validate generated tiles."""
    results = {
        "valid": True,
        "issues": [],
        "stats": {},
    }
    
    tiles_path = Path(tiles_dir)
    
    if not tiles_path.exists():
        results["valid"] = False
        results["issues"].append(f"Tiles directory not found: {tiles_dir}")
        return results
    
    for split in ["train", "val", "test"]:
        images_dir = tiles_path / split / "images"
        masks_dir = tiles_path / split / "masks"
        
        if not images_dir.exists():
            results["issues"].append(f"Missing images directory for {split}")
            continue
        
        # Count and validate tiles
        image_files = list(images_dir.glob("*.tif"))
        mask_files = list(masks_dir.glob("*.tif")) if masks_dir.exists() else []
        
        results["stats"][f"{split}_images"] = len(image_files)
        results["stats"][f"{split}_masks"] = len(mask_files)
        
        # Check for size mismatches
        wrong_size_count = 0
        for f in image_files[:100]:  # Sample first 100
            try:
                with rasterio.open(f) as src:
                    if src.width != expected_size or src.height != expected_size:
                        wrong_size_count += 1
            except:
                pass
        
        if wrong_size_count > 0:
            results["issues"].append(
                f"{split}: Found tiles with incorrect size (expected {expected_size}x{expected_size})"
            )
            results["valid"] = False
        
        # Check image-mask pairing
        image_stems = {f.stem for f in image_files}
        mask_stems = {f.stem.replace("_mask", "") for f in mask_files}
        
        missing_masks = image_stems - mask_stems
        if missing_masks:
            results["issues"].append(
                f"{split}: {len(missing_masks)} images missing corresponding masks"
            )
    
    return results


def validate_pipeline(
    data_dir: str = "data",
    tiles_dir: str = "data/processed/tiles",
    expected_tile_size: int = 512,
) -> None:
    """Run full pipeline validation."""
    
    print("=" * 60)
    print("PIPELINE VALIDATION REPORT")
    print("=" * 60)
    
    # Validate manifest
    print("\n1. MANIFEST VALIDATION")
    print("-" * 40)
    manifest_path = Path(data_dir) / "manifest.csv"
    
    if manifest_path.exists():
        manifest_results = validate_manifest(str(manifest_path))
        print(f"   Total images: {manifest_results['stats'].get('total_rows', 0)}")
        print(f"   Cities: {manifest_results['stats'].get('cities', [])}")
        print(f"   Missing annotations: {manifest_results['stats'].get('missing_annotations', 0)}")
        print(f"   Missing images: {manifest_results['stats'].get('missing_images', 0)}")
        
        if manifest_results["issues"]:
            print("   Issues:")
            for issue in manifest_results["issues"]:
                print(f"   ⚠ {issue}")
        else:
            print("   ✓ No issues found")
    else:
        print(f"   ✗ Manifest not found: {manifest_path}")
    
    # Validate splits
    print("\n2. SPLITS VALIDATION")
    print("-" * 40)
    splits_dir = Path(data_dir) / "splits"
    
    if splits_dir.exists():
        splits_results = validate_splits(str(splits_dir), str(manifest_path))
        
        for split in ["train", "val", "test"]:
            count = splits_results["stats"].get(f"{split}_count", 0)
            print(f"   {split.capitalize()}: {count} samples")
        
        if splits_results["issues"]:
            print("   Issues:")
            for issue in splits_results["issues"]:
                print(f"   ⚠ {issue}")
        else:
            print("   ✓ No issues found")
    else:
        print(f"   ✗ Splits directory not found: {splits_dir}")
    
    # Validate tiles
    print("\n3. TILES VALIDATION")
    print("-" * 40)
    
    if Path(tiles_dir).exists():
        tiles_results = validate_tiles(tiles_dir, expected_tile_size)
        
        for split in ["train", "val", "test"]:
            images = tiles_results["stats"].get(f"{split}_images", 0)
            masks = tiles_results["stats"].get(f"{split}_masks", 0)
            print(f"   {split.capitalize()}: {images} images, {masks} masks")
        
        if tiles_results["issues"]:
            print("   Issues:")
            for issue in tiles_results["issues"]:
                print(f"   ⚠ {issue}")
        else:
            print("   ✓ No issues found")
    else:
        print(f"   ✗ Tiles directory not found: {tiles_dir}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="data/processed/tiles",
        help="Tiles directory",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected tile size",
    )
    
    args = parser.parse_args()
    
    validate_pipeline(
        args.data_dir,
        args.tiles_dir,
        args.expected_size,
    )


if __name__ == "__main__":
    main()

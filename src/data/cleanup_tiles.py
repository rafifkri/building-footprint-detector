"""
Cleanup script to remove incorrectly sized tiles from the processed data.
This removes full-size images that were accidentally copied to the tiles folder.
"""

import argparse
from pathlib import Path

import rasterio
from tqdm import tqdm


def cleanup_tiles(
    tiles_dir: str,
    expected_size: int = 512,
    dry_run: bool = True,
) -> dict:
    """
    Remove tiles that don't match the expected size.
    
    Args:
        tiles_dir: Base directory containing tiles (e.g., data/processed/tiles)
        expected_size: Expected tile size in pixels
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dictionary with cleanup statistics
    """
    tiles_path = Path(tiles_dir)
    
    stats = {
        "total_files": 0,
        "correct_size": 0,
        "wrong_size": 0,
        "deleted": 0,
        "errors": 0,
    }
    
    # Process all splits
    for split in ["train", "val", "test"]:
        for subdir in ["images", "masks"]:
            split_dir = tiles_path / split / subdir
            
            if not split_dir.exists():
                continue
            
            print(f"\nProcessing {split}/{subdir}...")
            
            files_to_delete = []
            
            for f in tqdm(list(split_dir.glob("*.tif")) + list(split_dir.glob("*.png"))):
                stats["total_files"] += 1
                
                try:
                    with rasterio.open(f) as src:
                        width, height = src.width, src.height
                    
                    if width == expected_size and height == expected_size:
                        stats["correct_size"] += 1
                    else:
                        stats["wrong_size"] += 1
                        files_to_delete.append((f, width, height))
                        
                except Exception as e:
                    print(f"Error reading {f}: {e}")
                    stats["errors"] += 1
            
            # Delete or report files
            for f, w, h in files_to_delete:
                if dry_run:
                    print(f"  Would delete: {f.name} ({w}x{h})")
                else:
                    try:
                        f.unlink()
                        stats["deleted"] += 1
                    except Exception as e:
                        print(f"  Error deleting {f}: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Cleanup incorrectly sized tiles")
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="data/processed/tiles",
        help="Base directory containing tiles",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected tile size in pixels",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry run)",
    )
    
    args = parser.parse_args()
    
    print(f"Tile cleanup utility")
    print(f"Directory: {args.tiles_dir}")
    print(f"Expected size: {args.expected_size}x{args.expected_size}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("=" * 50)
    
    stats = cleanup_tiles(
        args.tiles_dir,
        args.expected_size,
        dry_run=not args.execute,
    )
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total files scanned: {stats['total_files']}")
    print(f"  Correct size: {stats['correct_size']}")
    print(f"  Wrong size: {stats['wrong_size']}")
    
    if args.execute:
        print(f"  Deleted: {stats['deleted']}")
    else:
        print(f"  Would delete: {stats['wrong_size']} files")
        print("\nRun with --execute to actually delete files.")


if __name__ == "__main__":
    main()

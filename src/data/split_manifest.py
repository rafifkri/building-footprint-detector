"""
Split manifest into train/val/test sets with stratification by city.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_manifest(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "city",
    random_state: int = 42,
) -> dict:
    """
    Split manifest CSV into train/val/test sets.
    
    Args:
        manifest_path: Path to manifest CSV
        output_dir: Directory to save split CSVs
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify_by: Column to stratify by (usually 'city')
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with split DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    df = pd.read_csv(manifest_path)
    print(f"Total samples: {len(df)}")
    
    stratify_col = df[stratify_by] if stratify_by in df.columns else None
    
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=random_state,
    )
    
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    
    if stratify_by in df.columns:
        temp_stratify = temp_df[stratify_by]
    else:
        temp_stratify = None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_stratify,
        random_state=random_state,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    if stratify_by in df.columns:
        print(f"\nDistribution by {stratify_by}:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"  {split_name}:")
            for city, count in split_df[stratify_by].value_counts().items():
                print(f"    {city}: {count}")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Split manifest into train/val/test")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits",
        help="Output directory for split CSVs",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="city",
        help="Column to stratify by",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    split_manifest(
        args.manifest,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.stratify_by,
        args.seed,
    )


if __name__ == "__main__":
    main()

"""
Convert raster masks to vector polygons (GeoJSON).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

from src.postprocess.postproc import postprocess_mask


def mask_to_polygons(
    mask: np.ndarray,
    transform: rasterio.Affine = None,
    simplify_tolerance: float = 1.0,
    min_area: float = 0.0,
) -> List[Polygon]:
    """
    Convert binary mask to list of polygons.
    
    Args:
        mask: Binary mask array
        transform: Affine transform for georeferencing
        simplify_tolerance: Tolerance for polygon simplification
        min_area: Minimum polygon area to keep
        
    Returns:
        List of Shapely polygons
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    
    for contour in contours:
        if len(contour) < 4:
            continue
        
        coords = contour.squeeze()
        if coords.ndim != 2:
            continue
        
        if transform is not None:
            geo_coords = []
            for x, y in coords:
                geo_x, geo_y = transform * (x, y)
                geo_coords.append((geo_x, geo_y))
            coords = geo_coords
        else:
            coords = [(float(x), float(y)) for x, y in coords]
        
        if len(coords) < 4:
            continue
        
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            poly = Polygon(coords)
            
            if not poly.is_valid:
                poly = make_valid(poly)
            
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    if p.area >= min_area:
                        if simplify_tolerance > 0:
                            p = p.simplify(simplify_tolerance)
                        polygons.append(p)
            elif poly.area >= min_area:
                if simplify_tolerance > 0:
                    poly = poly.simplify(simplify_tolerance)
                polygons.append(poly)
        except Exception as e:
            print(f"Warning: Failed to create polygon: {e}")
            continue
    
    return polygons


def mask_to_geojson(
    mask_path: str,
    output_path: str,
    postprocess: bool = True,
    min_area: int = 100,
    simplify_tolerance: float = 1.0,
    target_crs: str = None,
) -> gpd.GeoDataFrame:
    """
    Convert raster mask to GeoJSON file.
    
    Args:
        mask_path: Path to input mask file
        output_path: Path to output GeoJSON file
        postprocess: Whether to apply post-processing
        min_area: Minimum building area in pixels
        simplify_tolerance: Tolerance for polygon simplification
        target_crs: Target CRS for output (e.g., 'EPSG:4326')
        
    Returns:
        GeoDataFrame with building polygons
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    
    if postprocess:
        mask = postprocess_mask(
            mask,
            min_area=min_area,
            min_hole_area=50,
            smooth=True,
        )
    
    polygons = mask_to_polygons(
        mask,
        transform=transform,
        simplify_tolerance=simplify_tolerance,
        min_area=min_area,
    )
    
    if not polygons:
        print("Warning: No polygons extracted from mask")
        gdf = gpd.GeoDataFrame({"geometry": []}, crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)
        
        gdf["area"] = gdf.geometry.area
        gdf["building_id"] = range(1, len(gdf) + 1)
    
    if target_crs and crs:
        gdf = gdf.to_crs(target_crs)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {len(gdf)} buildings to {output_path}")
    
    return gdf


def batch_mask_to_geojson(
    mask_dir: str,
    output_dir: str,
    postprocess: bool = True,
    min_area: int = 100,
    simplify_tolerance: float = 1.0,
    target_crs: str = None,
    merge_output: bool = False,
) -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """
    Convert multiple mask files to GeoJSON.
    
    Args:
        mask_dir: Directory containing mask files
        output_dir: Directory to save GeoJSON files
        postprocess: Whether to apply post-processing
        min_area: Minimum building area
        simplify_tolerance: Tolerance for simplification
        target_crs: Target CRS
        merge_output: Whether to merge all outputs into one file
        
    Returns:
        GeoDataFrame(s) with building polygons
    """
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mask_files = list(mask_path.glob("*.tif")) + list(mask_path.glob("*.tiff"))
    
    all_gdfs = {}
    
    for mask_file in mask_files:
        output_file = output_path / f"{mask_file.stem}.geojson"
        
        try:
            gdf = mask_to_geojson(
                str(mask_file),
                str(output_file),
                postprocess=postprocess,
                min_area=min_area,
                simplify_tolerance=simplify_tolerance,
                target_crs=target_crs,
            )
            all_gdfs[mask_file.stem] = gdf
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    if merge_output and all_gdfs:
        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs.values(), ignore_index=True),
            crs=list(all_gdfs.values())[0].crs,
        )
        merged.to_file(output_path / "merged_buildings.geojson", driver="GeoJSON")
        return merged
    
    return all_gdfs


def main():
    parser = argparse.ArgumentParser(description="Convert mask to GeoJSON")
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to input mask file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output GeoJSON file",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip post-processing",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum building area in pixels",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=1.0,
        help="Simplification tolerance",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default=None,
        help="Target CRS (e.g., EPSG:4326)",
    )
    
    args = parser.parse_args()
    
    mask_to_geojson(
        args.mask,
        args.output,
        postprocess=not args.no_postprocess,
        min_area=args.min_area,
        simplify_tolerance=args.simplify,
        target_crs=args.target_crs,
    )


if __name__ == "__main__":
    main()

"""
Test mask to GeoJSON conversion.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.postprocess.postproc import (
    postprocess_mask,
    remove_small_objects,
    remove_holes,
    smooth_boundaries,
)
from src.postprocess.mask_to_geojson import mask_to_polygons, mask_to_geojson


@pytest.fixture
def sample_mask():
    """Create a sample mask for testing."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    
    mask[100:200, 100:200] = 255
    mask[300:400, 300:400] = 255
    mask[50:55, 50:55] = 255
    
    return mask


def test_remove_small_objects(sample_mask):
    """Test small object removal."""
    cleaned = remove_small_objects(sample_mask, min_area=100)
    
    assert cleaned[150, 150] == 255
    assert cleaned[350, 350] == 255
    assert cleaned[52, 52] == 0


def test_remove_holes():
    """Test hole removal."""
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    mask[40:60, 40:60] = 0
    
    filled = remove_holes(mask, min_hole_area=500)
    
    assert filled[50, 50] == 255


def test_smooth_boundaries(sample_mask):
    """Test boundary smoothing."""
    smoothed = smooth_boundaries(sample_mask, kernel_size=3)
    
    assert smoothed.shape == sample_mask.shape
    assert smoothed.dtype == sample_mask.dtype


def test_postprocess_mask(sample_mask):
    """Test full post-processing pipeline."""
    processed = postprocess_mask(
        sample_mask,
        min_area=100,
        min_hole_area=50,
        smooth=True,
    )
    
    assert processed.shape == sample_mask.shape


def test_mask_to_polygons(sample_mask):
    """Test polygon extraction from mask."""
    polygons = mask_to_polygons(sample_mask)
    
    assert len(polygons) >= 2
    
    for poly in polygons:
        assert poly.is_valid
        assert poly.area > 0


def test_mask_to_polygons_with_transform(sample_mask):
    """Test polygon extraction with transform."""
    transform = from_bounds(0, 0, 100, 100, 512, 512)
    
    polygons = mask_to_polygons(sample_mask, transform=transform)
    
    assert len(polygons) >= 2


def test_mask_to_geojson():
    """Test full mask to GeoJSON conversion."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:100, 50:100] = 255
    mask[150:200, 150:200] = 255
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = Path(tmpdir) / "mask.tif"
        output_path = Path(tmpdir) / "buildings.geojson"
        
        transform = from_bounds(0, 0, 256, 256, 256, 256)
        
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=1,
            dtype="uint8",
            transform=transform,
            crs="EPSG:4326",
        ) as dst:
            dst.write(mask, 1)
        
        gdf = mask_to_geojson(
            str(mask_path),
            str(output_path),
            postprocess=False,
            min_area=10,
        )
        
        assert output_path.exists()
        assert len(gdf) == 2


def test_empty_mask():
    """Test handling of empty mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    polygons = mask_to_polygons(mask)
    
    assert len(polygons) == 0


def test_full_mask():
    """Test handling of fully filled mask."""
    mask = np.ones((256, 256), dtype=np.uint8) * 255
    
    polygons = mask_to_polygons(mask)
    
    assert len(polygons) == 1

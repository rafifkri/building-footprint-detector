"""
Post-processing utilities for mask cleanup.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


def remove_small_objects(
    mask: np.ndarray,
    min_area: int = 100,
) -> np.ndarray:
    """
    Remove small objects from binary mask.
    
    Args:
        mask: Binary mask (0 or 255)
        min_area: Minimum area in pixels to keep
        
    Returns:
        Cleaned mask
    """
    binary = (mask > 0).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    cleaned = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    
    return cleaned


def remove_holes(
    mask: np.ndarray,
    min_hole_area: int = 50,
) -> np.ndarray:
    """
    Remove small holes from binary mask.
    
    Args:
        mask: Binary mask (0 or 255)
        min_hole_area: Minimum hole area to keep
        
    Returns:
        Mask with holes filled
    """
    binary = (mask > 0).astype(np.uint8)
    
    inverted = 1 - binary
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    
    filled = binary.copy()
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_hole_area:
            filled[labels == i] = 1
    
    return filled * 255


def smooth_boundaries(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Smooth mask boundaries using morphological operations.
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
        iterations: Number of iterations
        
    Returns:
        Smoothed mask
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=iterations
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )
    
    return closed


def separate_touching_objects(
    mask: np.ndarray,
    min_distance: int = 10,
) -> np.ndarray:
    """
    Separate touching objects using watershed.
    
    Args:
        mask: Binary mask
        min_distance: Minimum distance between object centers
        
    Returns:
        Separated mask
    """
    binary = (mask > 0).astype(np.uint8)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    local_max = ndimage.maximum_filter(dist_transform, size=min_distance)
    markers = (dist_transform == local_max) & (dist_transform > 0)
    
    markers = ndimage.label(markers)[0]
    
    markers = markers + 1
    markers[binary == 0] = 0
    
    rgb = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    
    markers = cv2.watershed(rgb, markers)
    
    separated = np.zeros_like(mask)
    separated[markers > 1] = 255
    
    return separated


def postprocess_mask(
    mask: np.ndarray,
    min_area: int = 100,
    min_hole_area: int = 50,
    smooth: bool = True,
    kernel_size: int = 3,
    separate_touching: bool = False,
) -> np.ndarray:
    """
    Apply full post-processing pipeline to mask.
    
    Args:
        mask: Input binary mask
        min_area: Minimum object area to keep
        min_hole_area: Minimum hole area to keep
        smooth: Whether to smooth boundaries
        kernel_size: Kernel size for smoothing
        separate_touching: Whether to separate touching objects
        
    Returns:
        Post-processed mask
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    result = remove_small_objects(mask, min_area)
    
    result = remove_holes(result, min_hole_area)
    
    if smooth:
        result = smooth_boundaries(result, kernel_size)
    
    if separate_touching:
        result = separate_touching_objects(result)
    
    return result

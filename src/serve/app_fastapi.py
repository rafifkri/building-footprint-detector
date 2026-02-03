"""
FastAPI service for building footprint detection.
"""

import io
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_smp import load_model_from_checkpoint
from src.training.inference import TiledInference
from src.postprocess.postproc import postprocess_mask
from src.postprocess.mask_to_geojson import mask_to_polygons
from src.utils.config import load_config


# Global variables for model
model = None
config = None
inferencer = None


def load_model_on_startup():
    """Load model on startup."""
    global model, config, inferencer
    
    config_path = "configs/infer.yaml"
    checkpoint_path = "checkpoints/best.pth"
    
    if not Path(config_path).exists():
        print(f"Warning: Config file not found: {config_path}")
        return
    
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        config = load_config(config_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_config = config.get("model", {})
        model = load_model_from_checkpoint(checkpoint_path, model_config, device)
        
        infer_config = config.get("inference", {})
        tta_config = config.get("tta", {})
        
        inferencer = TiledInference(
            model=model,
            device=device,
            tile_size=infer_config.get("tile_size", 512),
            overlap=infer_config.get("overlap", 128),
            threshold=infer_config.get("threshold", 0.5),
            use_tta=tta_config.get("enabled", True),
        )
        
        print(f"Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    load_model_on_startup()
    yield
    # Cleanup on shutdown
    global model, config, inferencer
    model = None
    config = None
    inferencer = None


app = FastAPI(
    title="Building Footprint Detection API",
    description="API for detecting building footprints in satellite imagery",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceSettings(BaseModel):
    tile_size: int = 512
    overlap: int = 128
    threshold: float = 0.5
    use_tta: bool = True
    min_area: int = 100
    smooth: bool = True


class DetectionResponse(BaseModel):
    num_buildings: int
    buildings: List[dict]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Building Footprint Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(...),
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
    use_tta: bool = True,
    min_area: int = 100,
    smooth: bool = True,
):
    """
    Predict building footprints from uploaded image.
    
    Args:
        file: Uploaded image file
        tile_size: Size of tiles for inference
        overlap: Overlap between tiles
        threshold: Threshold for binary mask
        use_tta: Whether to use test-time augmentation
        min_area: Minimum building area
        smooth: Whether to smooth boundaries
        
    Returns:
        Detection results with building polygons
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        local_inferencer = TiledInference(
            model=model,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            use_tta=use_tta,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            import rasterio
            from rasterio.transform import from_bounds
            
            height, width = image_np.shape[:2]
            transform = from_bounds(0, 0, width, height, width, height)
            
            with rasterio.open(
                tmp.name,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=3,
                dtype=image_np.dtype,
                transform=transform,
            ) as dst:
                for i in range(3):
                    dst.write(image_np[:, :, i], i + 1)
            
            mask, _ = local_inferencer.predict(tmp.name)
        
        mask = postprocess_mask(
            mask,
            min_area=min_area,
            smooth=smooth,
        )
        
        polygons = mask_to_polygons(mask)
        
        from shapely.geometry import mapping
        
        buildings = []
        for i, poly in enumerate(polygons):
            buildings.append({
                "id": i,
                "geometry": mapping(poly),
                "area": poly.area,
            })
        
        return DetectionResponse(
            num_buildings=len(buildings),
            buildings=buildings,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/mask")
async def predict_mask(
    file: UploadFile = File(...),
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
    use_tta: bool = True,
    min_area: int = 100,
    smooth: bool = True,
):
    """
    Predict building footprints and return mask as PNG.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        local_inferencer = TiledInference(
            model=model,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            use_tta=use_tta,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            import rasterio
            from rasterio.transform import from_bounds
            
            height, width = image_np.shape[:2]
            transform = from_bounds(0, 0, width, height, width, height)
            
            with rasterio.open(
                tmp.name,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=3,
                dtype=image_np.dtype,
                transform=transform,
            ) as dst:
                for i in range(3):
                    dst.write(image_np[:, :, i], i + 1)
            
            mask, _ = local_inferencer.predict(tmp.name)
        
        mask = postprocess_mask(
            mask,
            min_area=min_area,
            smooth=smooth,
        )
        
        mask_image = Image.fromarray(mask)
        
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=mask.png"},
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/geojson")
async def predict_geojson(
    file: UploadFile = File(...),
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
    use_tta: bool = True,
    min_area: int = 100,
    smooth: bool = True,
):
    """
    Predict building footprints and return GeoJSON.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        local_inferencer = TiledInference(
            model=model,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            use_tta=use_tta,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            import rasterio
            from rasterio.transform import from_bounds
            
            height, width = image_np.shape[:2]
            transform = from_bounds(0, 0, width, height, width, height)
            
            with rasterio.open(
                tmp.name,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=3,
                dtype=image_np.dtype,
                transform=transform,
            ) as dst:
                for i in range(3):
                    dst.write(image_np[:, :, i], i + 1)
            
            mask, _ = local_inferencer.predict(tmp.name)
        
        mask = postprocess_mask(
            mask,
            min_area=min_area,
            smooth=smooth,
        )
        
        polygons = mask_to_polygons(mask)
        
        from shapely.geometry import mapping
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "id": i,
                        "area": poly.area,
                    },
                }
                for i, poly in enumerate(polygons)
            ],
        }
        
        return JSONResponse(content=geojson)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Streamlit demo application for building footprint detection.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_smp import load_model_from_checkpoint
from src.training.inference import TiledInference
from src.postprocess.postproc import postprocess_mask
from src.postprocess.mask_to_geojson import mask_to_polygons
from src.utils.vis import create_mask_overlay, denormalize_image
from src.utils.config import load_config


st.set_page_config(
    page_title="Building Footprint Detection",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load model with caching."""
    config = load_config(config_path)
    model_config = config.get("model", {})
    model = load_model_from_checkpoint(checkpoint_path, model_config, device)
    return model, config


def process_image(
    image: np.ndarray,
    model: torch.nn.Module,
    device: str,
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
    use_tta: bool = True,
) -> np.ndarray:
    """Process a single image."""
    inferencer = TiledInference(
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
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        height, width = image.shape[:2]
        transform = from_bounds(0, 0, width, height, width, height)
        
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=image.dtype,
            transform=transform,
        ) as dst:
            for i in range(3):
                dst.write(image[:, :, i], i + 1)
        
        mask, _ = inferencer.predict(tmp.name)
    
    return mask


def main():
    st.title("Building Footprint Detection")
    st.markdown("Upload a satellite image to detect building footprints.")
    
    with st.sidebar:
        st.header("Settings")
        
        config_path = st.text_input(
            "Config Path",
            value="configs/infer.yaml",
        )
        
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoints/best.pth",
        )
        
        device = st.selectbox(
            "Device",
            options=["cuda", "cpu"],
            index=0 if torch.cuda.is_available() else 1,
        )
        
        st.subheader("Inference Settings")
        
        tile_size = st.slider(
            "Tile Size",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
        )
        
        overlap = st.slider(
            "Overlap",
            min_value=0,
            max_value=256,
            value=128,
            step=32,
        )
        
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        
        use_tta = st.checkbox("Use TTA", value=True)
        
        st.subheader("Post-processing")
        
        min_area = st.slider(
            "Min Building Area",
            min_value=0,
            max_value=500,
            value=100,
            step=10,
        )
        
        smooth = st.checkbox("Smooth Boundaries", value=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        
        uploaded_file = st.file_uploader(
            "Upload satellite image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
            
            st.image(image_np, caption="Uploaded Image", use_column_width=True)
            
            st.info(f"Image size: {image_np.shape[1]} x {image_np.shape[0]}")
    
    with col2:
        st.subheader("Prediction")
        
        if uploaded_file is not None:
            if st.button("Run Detection", type="primary"):
                try:
                    with st.spinner("Loading model..."):
                        model, config = load_model(
                            config_path, checkpoint_path, device
                        )
                    
                    with st.spinner("Processing image..."):
                        mask = process_image(
                            image_np,
                            model,
                            device,
                            tile_size=tile_size,
                            overlap=overlap,
                            threshold=threshold,
                            use_tta=use_tta,
                        )
                        
                        mask = postprocess_mask(
                            mask,
                            min_area=min_area,
                            smooth=smooth,
                        )
                    
                    overlay = create_mask_overlay(
                        image_np, mask, color=(255, 0, 0), alpha=0.4
                    )
                    
                    st.image(overlay, caption="Detection Result", use_column_width=True)
                    
                    polygons = mask_to_polygons(mask)
                    st.success(f"Detected {len(polygons)} buildings")
                    
                    st.session_state["mask"] = mask
                    st.session_state["polygons"] = polygons
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if "mask" in st.session_state:
            st.subheader("Export")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                mask_bytes = io.BytesIO()
                mask_img = Image.fromarray(st.session_state["mask"])
                mask_img.save(mask_bytes, format="PNG")
                
                st.download_button(
                    label="Download Mask (PNG)",
                    data=mask_bytes.getvalue(),
                    file_name="building_mask.png",
                    mime="image/png",
                )
            
            with col_b:
                import json
                from shapely.geometry import mapping
                
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": mapping(poly),
                            "properties": {"id": i},
                        }
                        for i, poly in enumerate(st.session_state["polygons"])
                    ],
                }
                
                st.download_button(
                    label="Download GeoJSON",
                    data=json.dumps(geojson, indent=2),
                    file_name="buildings.geojson",
                    mime="application/json",
                )
    
    st.markdown("---")
    st.markdown(
        "Building Footprint Detection using U-Net. "
        "Upload a satellite image to detect building footprints."
    )


if __name__ == "__main__":
    main()

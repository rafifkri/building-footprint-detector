# Building Footprint Detection Pipeline

End-to-end pipeline for building footprint detection using satellite imagery (SpaceNet-style).

## Project Overview

This project implements a complete workflow for detecting building footprints from satellite images:

- Data preprocessing (rasterization, tiling)
- Model training (PyTorch, U-Net via segmentation_models_pytorch)
- Evaluation and metrics
- Large-scale inference with tiling and TTA
- Post-processing and polygon export (GeoJSON)
- Demo applications (Streamlit/FastAPI)

## Project Structure

```
.
├── data/
│   ├── raw/                    # Original GeoTIFF/PNG + annotations
│   ├── manifest.csv            # Image-mask pairing
│   ├── splits/                 # Train/val/test splits
│   └── processed/              # Rasterized masks and tiles
├── configs/                    # Training and inference configs
├── src/
│   ├── data/                   # Preprocessing scripts
│   ├── models/                 # Model definitions and losses
│   ├── training/               # Train/eval/inference
│   ├── postprocess/            # Mask cleanup and vectorization
│   ├── utils/                  # Metrics, visualization, config
│   └── serve/                  # Demo applications
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs
├── notebooks/                  # EDA and experiments
├── demo/                       # Demo assets
└── tests/                      # Unit and integration tests
```

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate building-footprint
```

### Using Docker

```bash
docker build -t building-footprint .
docker run -it --gpus all building-footprint
```

## Quick Start

### 1. Prepare Data

```bash
# Build manifest from raw data
python -m src.data.build_manifest --input data/raw --output data/manifest.csv

# Create train/val/test splits
python -m src.data.split_manifest --manifest data/manifest.csv --output data/splits

# Rasterize vector annotations to masks
python -m src.data.rasterize_annotations --manifest data/manifest.csv --output data/processed/full_masks

# Create tiles for training
python -m src.data.tile_images --manifest data/manifest.csv --masks data/processed/full_masks --output data/processed/tiles
```

### 2. Train Model

```bash
python -m src.training.train --config configs/train.yaml
```

### 3. Evaluate Model

```bash
python -m src.training.eval --config configs/train.yaml --checkpoint checkpoints/best.pth
```

### 4. Run Inference

```bash
python -m src.training.inference --config configs/infer.yaml --input path/to/image.tif --output output/
```

### 5. Export to GeoJSON

```bash
python -m src.postprocess.mask_to_geojson --mask output/pred_mask.tif --output output/buildings.geojson
```

## Demo

### Streamlit App

```bash
streamlit run src/serve/app_streamlit.py
```

### FastAPI Service

```bash
uvicorn src.serve.app_fastapi:app --host 0.0.0.0 --port 8000
```

## Configuration

All configurations are stored in YAML files under `configs/`:

- `train.yaml`: Training hyperparameters
- `infer.yaml`: Inference settings
- `experiments/`: Experiment-specific configs

## Metrics

- IoU (Intersection over Union)
- Dice Coefficient
- Precision / Recall / F1

## Data Format

### Input

- Images: GeoTIFF or PNG (RGB or multispectral)
- Annotations: GeoJSON polygons

### Output

- Predicted masks: GeoTIFF (single-band binary)
- Vectorized polygons: GeoJSON

## Cities Supported

- Khartoum
- Paris
- Shanghai
- Vegas

## Acknowledgments

- SpaceNet dataset
- segmentation_models_pytorch

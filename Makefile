.PHONY: all clean install test train eval infer demo-streamlit demo-fastapi lint format

PYTHON = python
CONFIG = configs/train.yaml
INFER_CONFIG = configs/infer.yaml
CHECKPOINT = checkpoints/best.pth

all: install

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 mypy

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf build dist *.egg-info

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-smoke:
	pytest tests/test_inference_smoke.py -v

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

manifest:
	$(PYTHON) -m src.data.build_manifest --input data/raw --output data/manifest.csv

split:
	$(PYTHON) -m src.data.split_manifest --manifest data/manifest.csv --output data/splits

rasterize:
	$(PYTHON) -m src.data.rasterize_annotations --manifest data/manifest.csv --output data/processed/full_masks

tile:
	$(PYTHON) -m src.data.tile_images --manifest data/manifest.csv --masks data/processed/full_masks --output data/processed/tiles

preprocess: manifest split rasterize tile

train:
	$(PYTHON) -m src.training.train --config $(CONFIG)

eval:
	$(PYTHON) -m src.training.eval --config $(CONFIG) --checkpoint $(CHECKPOINT)

infer:
	$(PYTHON) -m src.training.inference --config $(INFER_CONFIG)

demo-streamlit:
	streamlit run src/serve/app_streamlit.py

demo-fastapi:
	uvicorn src.serve.app_fastapi:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t building-footprint .

docker-run:
	docker run -it --gpus all -v $(PWD)/data:/app/data -v $(PWD)/checkpoints:/app/checkpoints building-footprint

tensorboard:
	tensorboard --logdir logs/

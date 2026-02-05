"""Training utilities."""

# Lazy imports to avoid loading all dependencies upfront
def __getattr__(name):
    if name == "train":
        from .train import train
        return train
    elif name == "evaluate":
        from .eval import evaluate
        return evaluate
    elif name == "TiledInference":
        from .inference import TiledInference
        return TiledInference
    elif name == "run_inference":
        from .inference import run_inference
        return run_inference
    elif name == "BuildingDataset":
        from .dataset import BuildingDataset
        return BuildingDataset
    elif name == "InferenceDataset":
        from .dataset import InferenceDataset
        return InferenceDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "train",
    "evaluate",
    "TiledInference",
    "run_inference",
    "BuildingDataset",
    "InferenceDataset",
]

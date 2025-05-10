import importlib.util
import os

import torch
import importlib.util
import os

# Global cache for the model class
_model_class = None

def load_model_class(model_path: str, class_name: str):
    """Dynamically load a model class from a Python file and cache it."""
    global _model_class
    if _model_class is not None:
        return _model_class

    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ValueError(f"Class '{class_name}' not found in '{model_path}'")

    _model_class = getattr(module, class_name)
    return _model_class


def load_model_from_checkpoint(path: str, device="cpu") -> torch.nn.Module:
    """Instantiate the model and load weights from checkpoint."""
    if _model_class is None:
        raise RuntimeError("Model class not loaded. Call load_model_class first.")
    model = _model_class()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def contains_checkpoints(dir: str) -> list[str]:
    """Check for .pt or .ckpt files in the directory. Raise an error if none found."""
    try:
        files = os.listdir(dir)
    except Exception as e:
        raise Exception("error searching directory for model checkpoints")
    
    # Filter for checkpoint files
    checkpoint_files = [
        os.path.join(dir, f)
        for f in files
        if f.endswith(".pt") or f.endswith(".ckpt")
    ]

    if not checkpoint_files:
        raise Exception(f"could not find any valid model checkpoints in {os.getcwd()}")

    return checkpoint_files

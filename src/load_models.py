import importlib.util
import os
import inspect
import torch
import re
from typing import Dict, Any, Optional, Tuple, Type
from utils import logger
from collections.abc import Mapping

# Global cache for the model class
_model_class: Optional[Type] = None
_model_class_cache: Dict[Tuple[str, str], Type] = {}
# Cache for loaded models to avoid redundant loading
_model_cache: Dict[str, "torch.nn.Module"] = {}

import importlib.util
import os
import logging

logger = logging.getLogger(__name__)

# (abs_path, class_name) ➜ class  (keeps every distinct load)
_model_cache: Dict[Tuple[str, str], Type] = {}


def load_model_class(model_path: str, class_name: str):
    """
    Dynamically load (and cache) *class_name* from *model_path*.
    Keeps a per-file-and-name cache *and* sets `_model_class`
    so existing code that relies on that global continues to work.
    """
    global _model_class

    abs_path  = os.path.abspath(model_path)
    cache_key = (abs_path, class_name)

    # Fast-path: already loaded
    if cache_key in _model_class_cache:
        _model_class = _model_class_cache[cache_key]
        logger.debug("Model class already cached: %s from %s", class_name, abs_path)
        return _model_class

    # Load the module
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from path '{abs_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)        # type: ignore[arg-type]

    # Pull out the class
    if not hasattr(module, class_name):
        msg = f"Class '{class_name}' not found"
        logger.error(msg)
        raise ValueError(msg)

    cls = getattr(module, class_name)

    # Populate both caches
    _model_class_cache[cache_key] = cls
    _model_class = cls

    logger.info("Successfully loaded model class: %s", class_name)
    return cls



def extract_model_config_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the state dict to extract model configuration parameters.
    This function inspects layer dimensions and structure to infer the model architecture.
    """
    logger.debug("Extracting model configuration from state dict")
    config = {}
    
    # Pattern matching for common layer parameters
    linear_pattern = re.compile(r'(.+)\.weight$')
    
    # Analyze each key in the state dict
    for key, param in state_dict.items():
        # Extract hidden size from linear layers
        linear_match = linear_pattern.match(key)
        if linear_match and param.dim() == 2:
            layer_name = linear_match.group(1)
            
            # Common parameter extraction logic
            if layer_name == 'input_proj':
                config['input_dim'] = param.size(1)
                config['hidden_size'] = param.size(0)
            elif layer_name == 'output':
                config['output_dim'] = param.size(0)
            
            # Specific model architecture detection
            if 'blocks' in key:
                # Extract number of blocks based on highest index
                block_idx = int(key.split('.')[1])
                config['num_blocks'] = max(config.get('num_blocks', 0), block_idx + 1)
    
    #TODO: test this more throughly, dunno if this makes sense - that is, this entire function
    # Apply intelligent defaults for missing parameters
    if 'hidden_size' not in config:
        config['hidden_size'] = 128  # Reasonable default
        logger.debug("Using default hidden_size: 128")
    if 'num_blocks' not in config:
        config['num_blocks'] = 4     # Reasonable default
        logger.debug("Using default num_blocks: 4")
    if 'input_dim' not in config:
        config['input_dim'] = 10     # Reasonable default
        logger.debug("Using default input_dim: 10")
    
    logger.info(f"Extracted config: {config}")
    return config

#TODO: also check this with varying architectures
def initialize_model_with_config(model_class, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Initialize a model using the extracted configuration parameters.
    Tries multiple approaches to accommodate different model initialization signatures.
    """
    logger.debug(f"Initializing model with config: {config}")
    signature = inspect.signature(model_class.__init__)
    init_params = signature.parameters
    
    # Try different initialization approaches based on the model's __init__ signature
    try:
        # Approach 1: Try with config parameter if it exists
        if 'config' in init_params:
            logger.debug("Using config parameter approach")
            return model_class(config=config)
        
        # Approach 2: Try with explicit hidden_size, num_blocks parameters
        elif 'hidden_size' in init_params and 'num_blocks' in init_params:
            kwargs = {}
            for param in ['hidden_size', 'num_blocks', 'input_dim']:
                if param in init_params and param in config:
                    kwargs[param] = config[param]
            logger.debug(f"Using explicit parameters approach with kwargs: {kwargs}")
            return model_class(**kwargs)
        
        # Approach 3: Try with just hidden_size (common in many models)
        elif 'hidden_size' in init_params:
            logger.debug("Using hidden_size only approach")
            return model_class(hidden_size=config.get('hidden_size', 128))
        
        # Approach 4: Default initialization with no parameters
        else:
            logger.debug("Using default initialization approach")
            return model_class()
            
    except Exception as e:
        logger.warning(f"Error initializing model with extracted config: {e}")
        logger.info("Falling back to default initialization")
        # Fallback to default initialization
        return model_class()


def load_model_from_checkpoint(path: str, device: str = "auto") -> torch.nn.Module:
    global _model_cache, _model_class

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = f"{path}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if _model_class is None:
        raise RuntimeError("Model class not loaded. Call load_model_class first.")

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model_config = checkpoint.get("config")
    else:
        state_dict, model_config = checkpoint, None

    if model_config is None:
        model_config = extract_model_config_from_state_dict(state_dict)

    model = initialize_model_with_config(_model_class, model_config)

    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint does not contain a valid state dict")

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            model_keys = model.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_keys and v.shape == model_keys[k].shape}
            if not filtered:
                raise RuntimeError("Failed to load state dict into model")
            model.load_state_dict(filtered, strict=False)

    model.to(device)
    model.eval()
    _model_cache[cache_key] = model
    return model

def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    cache_size = len(_model_cache)
    _model_cache = {}
    logger.info(f"Cleared model cache ({cache_size} models)")

def contains_checkpoints(directory: str) -> list[str]:
    """
    Return a list of *.pt / *.ckpt* files in *directory*.

    Raises
    ------
    Exception
        If the directory cannot be read or contains no checkpoints.
    """
    logger.info("Searching for checkpoints in: %s", directory)

    # 1️⃣  Read directory
    try:
        files = os.listdir(directory)
        logger.debug("Found %d entries", len(files))
    except Exception as e:
        msg = "error searching directory for model checkpoints"
        logger.error("%s: %s", msg, e)
        raise Exception(msg) from e

    # 2️⃣  Filter for candidate files
    checkpoint_files = [
        os.path.join(directory, f) for f in files
        if f.endswith(".pt") or f.endswith(".ckpt")
    ]

    # 3️⃣  No matches → raise with precise message
    if not checkpoint_files:
        msg = f"could not find any valid model checkpoints in {directory}"
        logger.error(msg)
        raise Exception(msg)

    logger.info("Found %d checkpoint files", len(checkpoint_files))
    return checkpoint_files
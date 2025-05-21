import importlib.util
import os
import inspect
import torch
import logging
import re
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase-viz")

# Global cache for the model class
_model_class = None
# Cache for loaded models to avoid redundant loading
_model_cache = {}

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

def extract_model_config_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the state dict to extract model configuration parameters.
    This function inspects layer dimensions and structure to infer the model architecture.
    """
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
    
    # Apply intelligent defaults for missing parameters
    if 'hidden_size' not in config:
        config['hidden_size'] = 128  # Reasonable default
    if 'num_blocks' not in config:
        config['num_blocks'] = 4     # Reasonable default
    if 'input_dim' not in config:
        config['input_dim'] = 10     # Reasonable default
    
    return config

def initialize_model_with_config(model_class, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Initialize a model using the extracted configuration parameters.
    Tries multiple approaches to accommodate different model initialization signatures.
    """
    signature = inspect.signature(model_class.__init__)
    init_params = signature.parameters
    
    # Try different initialization approaches based on the model's __init__ signature
    try:
        # Approach 1: Try with config parameter if it exists
        if 'config' in init_params:
            return model_class(config=config)
        
        # Approach 2: Try with explicit hidden_size, num_blocks parameters
        elif 'hidden_size' in init_params and 'num_blocks' in init_params:
            kwargs = {}
            for param in ['hidden_size', 'num_blocks', 'input_dim']:
                if param in init_params and param in config:
                    kwargs[param] = config[param]
            return model_class(**kwargs)
        
        # Approach 3: Try with just hidden_size (common in many models)
        elif 'hidden_size' in init_params:
            return model_class(hidden_size=config.get('hidden_size', 128))
        
        # Approach 4: Default initialization with no parameters
        else:
            return model_class()
            
    except Exception as e:
        logger.warning(f"Error initializing model with extracted config: {e}")
        # Fallback to default initialization
        return model_class()

def load_model_from_checkpoint(path: str, device="auto") -> torch.nn.Module:
    """
    Intelligently load a model from a checkpoint, automatically determining
    the correct model configuration by analyzing the state dictionary.
    
    Args:
        path: Path to the checkpoint file
        device: Device to load the model on ('auto', 'cpu', 'cuda', or specific cuda device)
        
    Returns:
        The loaded model
    """
    global _model_cache, _model_class
    
    # Auto-detect device if set to auto
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check cache first
    cache_key = f"{path}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    if _model_class is None:
        raise RuntimeError("Model class not loaded. Call load_model_class first.")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
    
    # Extract the state dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the entire dict is the state dict if no specific keys found
            state_dict = checkpoint
    else:
        # If not a dict, assume the entire object is the state dict (unusual but possible)
        state_dict = checkpoint
    
    # Extract configuration from the state dict
    # First, check for explicit config in the checkpoint
    model_config = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        # If no explicit config, extract it from the state dict
        model_config = extract_model_config_from_state_dict(state_dict)
    
    # Create a model instance using the extracted configuration
    model = initialize_model_with_config(_model_class, model_config)
    
    # Load the state dict
    try:
        # Try loading with strict=True first (exact match)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        logger.warning(f"Strict loading failed, attempting non-strict loading: {e}")
        try:
            # If strict loading fails, try non-strict loading
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")
        except Exception as e2:
            # If both approaches fail, raise an error
            raise RuntimeError(f"Failed to load state dict into model: {e2}")
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Cache model
    _model_cache[cache_key] = model
    
    return model

def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    _model_cache = {}

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
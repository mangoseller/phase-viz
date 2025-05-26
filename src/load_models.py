import importlib.util
import os
import inspect
import torch
import re
from typing import Dict, Any, Optional, Tuple, Type
from utils import logger
from collections.abc import Mapping
import logging

_model_class: Optional[Type] = None
_model_class_cache: Dict[Tuple[str, str], Type] = {}
logger = logging.getLogger(__name__)
_model_cache: Dict[Tuple[str, str], Type] = {}

def load_model_class(model_path: str, class_name: str):
    """
    Dynamically load (and cache) *class_name* from *model_path*.
    Keeps a per-file-and-name cache *and* sets `_model_class`
    so existing code that relies on that global continues to work.
    """
    global _model_class

    abs_path = os.path.abspath(model_path)
    cache_key = (abs_path, class_name)

    if cache_key in _model_class_cache:
        _model_class = _model_class_cache[cache_key]
        logger.debug("Model class already cached: %s from %s", class_name, abs_path)
        return _model_class

    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from path '{abs_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        msg = f"Class '{class_name}' not found"
        logger.error(msg)
        raise ValueError(msg)

    cls = getattr(module, class_name)
    _model_class_cache[cache_key] = cls
    _model_class = cls

    logger.info("Successfully loaded model class: %s", class_name)
    return cls


def extract_model_config_from_class(model_class: Type) -> Dict[str, Any]:
    """
    Extract model configuration by inspecting the model class's __init__ method.
    This is more reliable than trying to reverse-engineer from state dict.
    """
    logger.debug(f"Extracting model configuration from class: {model_class.__name__}")
    
    # Get the signature of the __init__ method
    signature = inspect.signature(model_class.__init__)
    
    # Extract parameter names and their defaults
    config = {}
    for param_name, param in signature.parameters.items():
        if param_name == 'self':
            continue
            
        # If parameter has a default value, use it
        if param.default != inspect.Parameter.empty:
            config[param_name] = param.default
        else:
            # For required parameters without defaults, we'll need to infer from state dict
            logger.debug(f"Parameter {param_name} has no default value")
    
    logger.info(f"Extracted config from class: {config}")
    return config


def infer_missing_config_from_state_dict(state_dict: Dict[str, Any], 
                                        model_class: Type,
                                        existing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    For any required parameters without defaults, try to infer them from the state dict.
    This is only used as a fallback for parameters we couldn't get from the class definition.
    """
    config = existing_config.copy()
    signature = inspect.signature(model_class.__init__)
    
    # Get list of required parameters (no defaults)
    required_params = []
    for param_name, param in signature.parameters.items():
        if param_name != 'self' and param.default == inspect.Parameter.empty:
            if param_name not in config:
                required_params.append(param_name)
    
    if not required_params:
        return config
    
    logger.debug(f"Attempting to infer required parameters: {required_params}")
    
    # Try to create a temporary model instance to see what parameters it expects
    # This is done by analyzing the state dict keys and dimensions
    
    for param_name in required_params:
        if param_name == 'input_dim' or param_name == 'input_size':
            # Look for the first layer's input dimension
            for key, tensor in state_dict.items():
                if 'weight' in key and tensor.dim() == 2:
                    if any(name in key for name in ['input', 'fc1', 'layer1', 'embed']):
                        config[param_name] = tensor.size(1)
                        logger.debug(f"Inferred {param_name} = {tensor.size(1)} from {key}")
                        break
                    elif key.endswith('.0.weight') or key == 'weight':
                        config[param_name] = tensor.size(1)
                        logger.debug(f"Inferred {param_name} = {tensor.size(1)} from {key}")
                        break
        
        elif param_name == 'output_dim' or param_name == 'output_size':
            # Look for the last layer's output dimension
            output_candidates = []
            for key, tensor in state_dict.items():
                if 'weight' in key and tensor.dim() == 2:
                    if any(name in key for name in ['output', 'head', 'fc_out']):
                        output_candidates.append((key, tensor.size(0)))
            
            if output_candidates:
                # Use the one with 'output' in the name if available
                for key, dim in output_candidates:
                    if 'output' in key:
                        config[param_name] = dim
                        logger.debug(f"Inferred {param_name} = {dim} from {key}")
                        break
                else:
                    # Otherwise use the first candidate
                    config[param_name] = output_candidates[0][1]
                    logger.debug(f"Inferred {param_name} = {output_candidates[0][1]} from {output_candidates[0][0]}")
        
        elif param_name == 'hidden_size' or param_name == 'hidden_dim':
            # Already handled by extract_model_config_from_class defaults usually
            pass
        
        elif param_name == 'vocab_size':
            # Look for embedding layers
            for key, tensor in state_dict.items():
                if 'embedding' in key and 'weight' in key and tensor.dim() == 2:
                    config[param_name] = tensor.size(0)
                    logger.debug(f"Inferred {param_name} = {tensor.size(0)} from {key}")
                    break
        
        elif param_name == 'd_model':
            # Look for transformer-specific patterns
            for key, tensor in state_dict.items():
                if 'embedding' in key and 'weight' in key and tensor.dim() == 2:
                    config[param_name] = tensor.size(1)
                    logger.debug(f"Inferred {param_name} = {tensor.size(1)} from {key}")
                    break
        
        elif param_name == 'num_layers':
            # Count layers by looking at keys
            layer_indices = set()
            for key in state_dict.keys():
                # Look for patterns like layer.0, layers.1, etc.
                matches = re.findall(r'(?:layers?|blocks?)\.(\d+)\.', key)
                for match in matches:
                    layer_indices.add(int(match))
            
            if layer_indices:
                config[param_name] = max(layer_indices) + 1
                logger.debug(f"Inferred {param_name} = {config[param_name]} from layer indices")
        
        elif param_name == 'num_blocks':
            # Similar to num_layers but specifically for blocks
            block_indices = set()
            for key in state_dict.keys():
                matches = re.findall(r'blocks\.(\d+)\.', key)
                for match in matches:
                    block_indices.add(int(match))
            
            if block_indices:
                config[param_name] = max(block_indices) + 1
                logger.debug(f"Inferred {param_name} = {config[param_name]} from block indices")
    
    # Log any parameters we couldn't infer
    missing = [p for p in required_params if p not in config]
    if missing:
        logger.warning(f"Could not infer required parameters: {missing}. Model initialization may fail.")
    
    return config


def initialize_model_with_config(model_class: Type, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Initialize a model using the configuration parameters.
    First tries to use the exact parameters, then falls back to different initialization patterns.
    """
    logger.debug(f"Initializing model {model_class.__name__} with config: {config}")
    
    # First, try to initialize with the exact config we have
    signature = inspect.signature(model_class.__init__)
    
    # Filter config to only include parameters that the model accepts
    init_params = {}
    for param_name, param in signature.parameters.items():
        if param_name != 'self' and param_name in config:
            init_params[param_name] = config[param_name]
    
    try:
        # Try direct initialization with filtered parameters
        model = model_class(**init_params)
        logger.debug(f"Successfully initialized model with parameters: {init_params}")
        return model
    except Exception as e:
        logger.debug(f"Direct initialization failed: {e}")
    
    # If that fails, try some common patterns
    
    # Pattern 1: Model expects a config dict
    if 'config' in signature.parameters:
        try:
            model = model_class(config=config)
            logger.debug("Successfully initialized model with config dict")
            return model
        except Exception as e:
            logger.debug(f"Config dict initialization failed: {e}")
    
    # Pattern 2: Try with just the defaults (no arguments)
    try:
        model = model_class()
        logger.debug("Successfully initialized model with defaults only")
        return model
    except Exception as e:
        logger.debug(f"Default initialization failed: {e}")
    
    # Pattern 3: Try with only the most common parameters
    common_params = ['hidden_size', 'input_dim', 'output_dim', 'd_model', 'num_layers']
    minimal_config = {k: v for k, v in config.items() if k in common_params and k in signature.parameters}
    
    if minimal_config:
        try:
            model = model_class(**minimal_config)
            logger.debug(f"Successfully initialized model with minimal config: {minimal_config}")
            return model
        except Exception as e:
            logger.debug(f"Minimal config initialization failed: {e}")
    
    # If all else fails, raise an informative error
    raise RuntimeError(
        f"Failed to initialize {model_class.__name__}. "
        f"Available config: {config}, "
        f"Model parameters: {list(signature.parameters.keys())}"
    )


def load_model_from_checkpoint(path: str, device: str = "auto") -> torch.nn.Module:
    """Load a model from a checkpoint file."""
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

    # Extract state dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        # Try different keys where state dict might be stored
        state_dict = None
        for key in ['model_state', 'state_dict', 'model_state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        # If no specific key found, assume the checkpoint itself is the state dict
        if state_dict is None:
            # But first check if it looks like a state dict (has tensor values)
            if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise RuntimeError(f"Could not find state dict in checkpoint. Keys found: {list(checkpoint.keys())}")
    else:
        # Checkpoint is directly the state dict
        state_dict = checkpoint

    # Get base configuration from the model class
    config = extract_model_config_from_class(_model_class)
    
    # If there's a config in the checkpoint, use it to override/update our extracted config
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        logger.debug("Found config in checkpoint, merging with extracted config")
        config.update(checkpoint['config'])
    
    # Try to infer any missing required parameters from the state dict
    config = infer_missing_config_from_state_dict(state_dict, _model_class, config)
    
    # Initialize the model
    model = initialize_model_with_config(_model_class, config)

    # Load the state dict
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Checkpoint does not contain a valid state dict")

    try:
        model.load_state_dict(state_dict, strict=True)
        logger.debug("Loaded state dict with strict=True")
    except Exception as e:
        logger.warning(f"Strict loading failed: {e}")
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.debug("Loaded state dict with strict=False")
        except Exception as e2:
            logger.warning(f"Non-strict loading also failed: {e2}")
            # As a last resort, try to load only matching keys
            model_keys = set(model.state_dict().keys())
            state_keys = set(state_dict.keys())
            
            # Find matching keys
            matching_keys = model_keys.intersection(state_keys)
            if not matching_keys:
                raise RuntimeError(f"No matching keys between model and checkpoint. Model keys: {list(model_keys)[:5]}..., Checkpoint keys: {list(state_keys)[:5]}...")
            
            # Load only matching keys
            filtered_dict = {k: state_dict[k] for k in matching_keys}
            model.load_state_dict(filtered_dict, strict=False)
            logger.warning(f"Loaded {len(matching_keys)}/{len(model_keys)} keys from checkpoint")

    model.to(device)
    model.eval()
    
    # Cache the loaded model
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
    try:
        files = os.listdir(directory)
        logger.debug("Found %d entries", len(files))
    except Exception as e:
        msg = "error searching directory for model checkpoints"
        logger.error("%s: %s", msg, e)
        raise Exception(msg) from e
    
    checkpoint_files = [
        os.path.join(directory, f) for f in files
        if f.endswith(".pt") or f.endswith(".ckpt")
    ]
    
    if not checkpoint_files:
        msg = f"could not find any valid model checkpoints in {directory}"
        logger.error(msg)
        raise Exception(msg)

    logger.info("Found %d checkpoint files", len(checkpoint_files))
    return checkpoint_files


# Export the new functions
__all__ = [
    'load_model_class',
    'extract_model_config_from_class',
    'infer_missing_config_from_state_dict',
    'initialize_model_with_config',
    'load_model_from_checkpoint',
    'clear_model_cache',
    'contains_checkpoints'
]
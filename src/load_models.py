import importlib.util
import os
import inspect
import torch
import re
from typing import Dict, Any, Optional, Tuple, Type, List
from utils import logger
from collections.abc import Mapping
import logging
import re

_model_class: Optional[Type] = None
_model_class_cache: Dict[Tuple[str, str], Type] = {}
logger = logging.getLogger(__name__)
_model_cache: Dict[Tuple[str, str], Type] = {}

def load_model_class(model_path: str, class_name: str):
    """
    Dynamically load (and cache) *class_name* from *model_path*.
    Keeps a per-file-and-name cache *and* sets `_model_class`
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
    Attempt to extract model configuration by inspecting the model class's __init__ method
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
            logger.debug(f"Parameter {param_name} has no default value")
    
    logger.info(f"Extracted config from class: {config}")
    return config


def infer_missing_config_from_state_dict(state_dict: Dict[str, Any], 
                                        model_class: Type,
                                        existing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    For any required parameters without defaults, attempt to infer required params from state_dict
    fallback for extract_model_config_from_class

    """
    config = existing_config.copy()
    signature = inspect.signature(model_class.__init__)
    
    # Get list of required parameters without defaults
    required_params = []
    for param_name, param in signature.parameters.items():
        if param_name != 'self' and param.default == inspect.Parameter.empty:
            if param_name not in config:
                required_params.append(param_name)
    
    if not required_params:
        return config
    
    logger.debug(f"Attempting to infer required parameters: {required_params}")
    
    for param_name in required_params:
        # Special handling for 'dims' parameter (list of layer dimensions)
        if param_name == 'dims':
            dims = infer_dims_from_state_dict(state_dict)
            if dims:
                config[param_name] = dims
                logger.debug(f"Inferred {param_name} = {dims} from layer shapes")
                continue
                
        elif param_name == 'input_dim' or param_name == 'input_size':
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
                for key, dim in output_candidates:
                    if 'output' in key:
                        config[param_name] = dim
                        logger.debug(f"Inferred {param_name} = {dim} from {key}")
                        break
                else:
                    config[param_name] = output_candidates[0][1]
                    logger.debug(f"Inferred {param_name} = {output_candidates[0][1]} from {output_candidates[0][0]}")
        
        elif param_name == 'hidden_size' or param_name == 'hidden_dim':
            pass
        
        elif param_name == 'vocab_size':
            for key, tensor in state_dict.items():
                if 'embedding' in key and 'weight' in key and tensor.dim() == 2:
                    config[param_name] = tensor.size(0)
                    logger.debug(f"Inferred {param_name} = {tensor.size(0)} from {key}")
                    break
        
        elif param_name == 'd_model':
            # Handle transformers
            for key, tensor in state_dict.items():
                if 'embedding' in key and 'weight' in key and tensor.dim() == 2:
                    config[param_name] = tensor.size(1)
                    logger.debug(f"Inferred {param_name} = {tensor.size(1)} from {key}")
                    break
        
        elif param_name == 'num_layers':
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
            # Handle block models, e.g. Resnets
            block_indices = set()
            for key in state_dict.keys():
                matches = re.findall(r'blocks\.(\d+)\.', key)
                for match in matches:
                    block_indices.add(int(match))
            
            if block_indices:
                config[param_name] = max(block_indices) + 1
                logger.debug(f"Inferred {param_name} = {config[param_name]} from block indices")
    
    # Log failures
    missing = [p for p in required_params if p not in config]
    if missing:
        logger.warning(f"Could not infer required parameters: {missing}. Model initialization may fail.")
    
    return config


def infer_dims_from_state_dict(state_dict: Dict[str, Any]) -> Optional[List[int]]:
    """
    Infer the 'dims' parameter (list of layer dimensions) from state dict.
    This handles models like DLN that use a list of dimensions to define architecture.
    """
    # Look for sequential linear layers patterns
    linear_patterns = [
        (r'linears\.(\d+)\.weight', 'linears'),  # DLN 
        (r'layers\.(\d+)\.weight', 'layers'),    # Common 
        (r'fc(\d+)\.weight', 'fc'),              # get fc layers
        (r'linear(\d+)\.weight', 'linear'),      # linear1, linear2, etc.
    ]
    
    for pattern, prefix in linear_patterns:
        layers = {}
        for key, tensor in state_dict.items():
            match = re.match(pattern, key)
            if match and tensor.dim() == 2:
                idx = int(match.group(1))
                layers[idx] = tensor
        
        if layers:
            # Sort by index
            sorted_indices = sorted(layers.keys())
            
            # Check if indices are contiguous
            if sorted_indices == list(range(len(sorted_indices))):
                dims = []
                
                # Get input dimension from first layer
                if 0 in layers:
                    dims.append(layers[0].size(1))  # Input size
                
                # Get output dimension of each layer
                for idx in sorted_indices:
                    dims.append(layers[idx].size(0))  # Output size
                
                if len(dims) >= 2:  # Need at least input and output
                    logger.debug(f"Inferred dims from {prefix} pattern: {dims}")
                    return dims
    

    weight_keys = [(k, v) for k, v in state_dict.items() 
                   if 'weight' in k and isinstance(v, torch.Tensor) and v.dim() == 2]
    
    if weight_keys:
        weight_keys.sort(key=lambda x: x[0])
        
        dims = []
        prev_out = None
        
        for key, tensor in weight_keys:
            in_dim, out_dim = tensor.size(1), tensor.size(0)
            
            if not dims:
                dims.extend([in_dim, out_dim])
                prev_out = out_dim
            elif in_dim == prev_out:
                dims.append(out_dim)
                prev_out = out_dim
            else:
                dims = []
                break
        
        if len(dims) >= 2:
            logger.debug(f"Inferred dims from weight chain: {dims}")
            return dims
    
    return None


def initialize_model_with_config(model_class: Type, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Initialize a model using the configuration parameters.
    First tries to use the exact parameters, then falls back to different initialization patterns.
    """
    logger.debug(f"Initializing model {model_class.__name__} with config: {config}")
    
    # attempt initialize with the exact config we have
    signature = inspect.signature(model_class.__init__)
    
    # try removing failed params
    init_params = {}
    for param_name, param in signature.parameters.items():
        if param_name != 'self' and param_name in config:
            init_params[param_name] = config[param_name]
    
    try:

        model = model_class(**init_params)
        logger.debug(f"Successfully initialized model with parameters: {init_params}")
        return model
    except Exception as e:
        logger.debug(f"Direct initialization failed: {e}")
    

    # TODO: remove these, get more broad compatibility to work
    # Fallback pattens:
  
    
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
    common_params = ['hidden_size', 'input_dim', 'output_dim', 'd_model', 'num_layers', 'dims']
    minimal_config = {k: v for k, v in config.items() if k in common_params and k in signature.parameters}
    
    if minimal_config:
        try:
            model = model_class(**minimal_config)
            logger.debug(f"Successfully initialized model with minimal config: {minimal_config}")
            return model
        except Exception as e:
            logger.debug(f"Minimal config initialization failed: {e}")
    
    if hasattr(model_class, 'make_rectangular') and all(k in config for k in ['gamma', 'w', 'L']):
        try:
            # Try to infer input/output dims from the inferred dims list
            if 'dims' in config and len(config['dims']) >= 2:
                input_dim = config['dims'][0]
                output_dim = config['dims'][-1]
                L = len(config['dims']) - 1
                w = config.get('w', config['dims'][1] if len(config['dims']) > 2 else 100)
                gamma = config.get('gamma', 1.0)
                
                model = model_class.make_rectangular(input_dim, output_dim, L, w, gamma)
                logger.debug(f"Successfully initialized model using make_rectangular factory method")
                return model
        except Exception as e:
            logger.debug(f"Factory method initialization failed: {e}")
    
    raise RuntimeError(
        f"Failed to initialize {model_class.__name__}. "
        f"Available config: {config}, "
        f"Model parameters: {list(signature.parameters.keys())}"
    )


def load_model_from_checkpoint(path: str, device: str = "auto") -> torch.nn.Module:
    """Load a model from a checkpoint file with enhanced configuration inference."""
    global _model_cache, _model_class

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = f"{path}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if _model_class is None:
        raise RuntimeError("Model class not loaded. Call load_model_class first.")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ['model_state', 'state_dict', 'model_state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is None:
            if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise RuntimeError(f"Could not find state dict in checkpoint. Keys found: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint

    config = extract_model_config_from_class(_model_class)
    
    # If there's a config in the checkpoint, use it to override/update our extracted config
    if isinstance(checkpoint, dict):
        # Look for config in various possible locations
        for config_key in ['config', 'model_config', 'hparams', 'hyper_parameters']:
            if config_key in checkpoint:
                logger.debug(f"Found config in checkpoint under '{config_key}', merging with extracted config")
                checkpoint_config = checkpoint[config_key]
                # Handle cases where config might be a namespace or other object
                if hasattr(checkpoint_config, '__dict__'):
                    checkpoint_config = checkpoint_config.__dict__
                if isinstance(checkpoint_config, dict):
                    config.update(checkpoint_config)
                break
    
    # Try to infer any missing required parameters from the state dict
    config = infer_missing_config_from_state_dict(state_dict, _model_class, config)
    
    # Special handling for DLN models TODO: remove this
    if _model_class.__name__ == 'DLN' and 'dims' not in config:
        # Try to infer dims from the model's stored attributes if they exist
        dims_from_attributes = try_infer_dims_from_attributes(state_dict)
        if dims_from_attributes:
            config['dims'] = dims_from_attributes
            logger.debug(f"Inferred dims from model attributes: {dims_from_attributes}")
    
    # Initialize the model
    try:
        model = initialize_model_with_config(_model_class, config)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        RuntimeError(f"{e}{suggestion}")
        raise 

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


def try_infer_dims_from_attributes(state_dict: Dict[str, Any]) -> Optional[List[int]]:
    """
    Try to infer dims from model attributes that might be stored in the state dict.
    Some models store configuration as buffer or attributes.
    """
    for key in ['dims', '_dims', 'layer_dims', '_layer_dims']:
        if key in state_dict:
            value = state_dict[key]
            if isinstance(value, torch.Tensor):
                dims = value.tolist()
                if isinstance(dims, list) and all(isinstance(d, (int, float)) for d in dims):
                    return [int(d) for d in dims]
            elif isinstance(value, list):
                return value
    
    # Look for individual dim attributes (input_dim, hidden_dims, output_dim)
    if 'input_dim' in state_dict and 'output_dim' in state_dict:
        input_dim = state_dict['input_dim']
        output_dim = state_dict['output_dim']
        
        if isinstance(input_dim, torch.Tensor):
            input_dim = input_dim.item()
        if isinstance(output_dim, torch.Tensor):
            output_dim = output_dim.item()
        
        # Check for hidden_dims
        hidden_dims = []
        if 'hidden_dims' in state_dict:
            hd = state_dict['hidden_dims']
            if isinstance(hd, torch.Tensor):
                hidden_dims = hd.tolist()
            elif isinstance(hd, list):
                hidden_dims = hd
        
        # Reconstruct dims
        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
        if len(dims) >= 2:
            return dims
    
    return None


def analyze_layer_structure(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze state dict to provide debugging info.
    """
    info = {
        'linear_layers': [],
        'other_layers': [],
        'buffers': [],
        'possible_dims': None
    }
    
    # Categorize keys
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if 'weight' in key and value.dim() == 2:
                info['linear_layers'].append({
                    'name': key,
                    'shape': list(value.shape),
                    'in_features': value.shape[1],
                    'out_features': value.shape[0]
                })
            elif 'weight' in key or 'bias' in key:
                info['other_layers'].append({
                    'name': key,
                    'shape': list(value.shape) if hasattr(value, 'shape') else 'scalar'
                })
        else:
            info['buffers'].append({
                'name': key,
                'type': type(value).__name__,
                'value': str(value)
            })
    
    # Try to infer possible dims from linear layers
    if info['linear_layers']:
        # Sort by name to get correct order
        info['linear_layers'].sort(key=lambda x: x['name'])
        
        # Try to extract a dims sequence
        dims_sequence = []
        for i, layer in enumerate(info['linear_layers']):
            if i == 0:
                dims_sequence.append(layer['in_features'])
            dims_sequence.append(layer['out_features'])
        
        info['possible_dims'] = dims_sequence
    
    return info


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    cache_size = len(_model_cache)
    _model_cache = {}
    logger.info(f"Cleared model cache ({cache_size} models)")




def natural_sort_key(text):
    """
    Generate a key for natural sorting that handles numbers properly.
    E.g., checkpoint_2.pt comes before checkpoint_10.pt
    """
    def convert(s):
        return int(s) if s.isdigit() else s
    
    # Split the string into alternating text and number chunks
    parts = re.split(r'(\d+)', text)
    # Convert number strings to actual integers for proper sorting
    return [convert(part) for part in parts]
    
def contains_checkpoints(directory: str) -> list[str]:

    # Return a list of *.pt / *.ckpt* files in *directory*, sorted naturally.

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

    checkpoint_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    logger.info("Found %d checkpoint files (sorted)", len(checkpoint_files))
    return checkpoint_files



__all__ = [
    'load_model_class',
    'extract_model_config_from_class',
    'infer_missing_config_from_state_dict',
    'initialize_model_with_config',
    'load_model_from_checkpoint',
    'clear_model_cache',
    'contains_checkpoints'
]
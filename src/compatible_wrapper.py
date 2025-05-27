import torch
import torch.nn as nn
from typing import Iterator, Tuple
import logging

logger = logging.getLogger(__name__)

class MetricsCompatibleModel(nn.Module):
    """
    Wrapper that makes any model compatible with standard metrics by:
    1. Creating composite 'weight' views for modules with multiple weight parameters
    2. Flattening 3D+ weight tensors to 2D when appropriate
    3. Creating 'weight' and 'bias' attributes for non-standard parameter names
    4. Preserving the original model structure
    """
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self._wrapped_params = {}
        self._composite_weights = {}
        self._setup_compatible_parameters()
    
    def _setup_compatible_parameters(self):
        """Create metric-compatible views of all parameters."""
        for name, module in self.original_model.named_modules():
            if module is self.original_model:
                continue
            
            # Collect all parameters in this module
            module_params = list(module.named_parameters(recurse=False))
            if not module_params:
                continue
                
            # Check what this module already has
            has_standard_weight = hasattr(module, 'weight') and module.weight is not None
            has_standard_bias = hasattr(module, 'bias') and module.bias is not None
            
            # Collect weight-like and bias-like parameters
            weight_params = []
            bias_params = []
            
            for param_name, param in module_params:
                if param is None:
                    continue
                
                # Identify weight-like parameters
                if not has_standard_weight and param.dim() >= 2:
                    # Check if this looks like a weight parameter
                    if any(indicator in param_name.upper() for indicator in ['W', 'WEIGHT', 'KERNEL']):
                        weight_params.append((param_name, param))
                    elif param_name in ['embedding', 'embed'] and param.dim() == 2:
                        weight_params.append((param_name, param))
                
                # Identify bias-like parameters  
                elif not has_standard_bias and param.dim() == 1:
                    if any(indicator in param_name.lower() for indicator in ['b', 'bias']):
                        bias_params.append((param_name, param))
            
            # Handle weight parameters
            if weight_params and not has_standard_weight:
                if len(weight_params) == 1:
                    # Single weight parameter - use it directly
                    param_name, param = weight_params[0]
                    if param.dim() == 2:
                        setattr(module, 'weight', param)
                    else:
                        # Flatten 3D+ tensors to 2D
                        flattened = param.view(-1, param.size(-1))
                        setattr(module, 'weight', flattened)
                else:
                    # Multiple weight parameters - create a composite view
                    # This concatenates all weight parameters into a single 2D tensor
                    composite_weight = self._create_composite_weight(module, weight_params)
                    if composite_weight is not None:
                        setattr(module, 'weight', composite_weight)
                        # Store reference for proper gradient handling
                        self._composite_weights[id(module)] = composite_weight
            
            # Handle bias parameters
            if bias_params and not has_standard_bias:
                if len(bias_params) == 1:
                    param_name, param = bias_params[0]
                    setattr(module, 'bias', param)
                else:
                    # Multiple bias parameters - concatenate them
                    composite_bias = self._create_composite_bias(module, bias_params)
                    if composite_bias is not None:
                        setattr(module, 'bias', composite_bias)
    
    def _create_composite_weight(self, module, weight_params):
        """Create a composite weight tensor from multiple weight parameters."""
        try:
            if not weight_params:
                return None
                
            # Strategy 1: If all weights have compatible dimensions, concatenate them
            # Check if all tensors have the same last dimension (common for attention)
            last_dims = [param.size(-1) for _, param in weight_params]
            if len(set(last_dims)) == 1:
                # All have same last dimension - can concatenate along first dims
                flattened_weights = []
                for param_name, param in weight_params:
                    if param.dim() == 2:
                        flattened_weights.append(param)
                    else:
                        # Flatten to 2D, preserving the last dimension
                        flat = param.view(-1, param.size(-1))
                        flattened_weights.append(flat)
                
                if flattened_weights:
                    composite = torch.cat(flattened_weights, dim=0)
                    logger.debug(f"Created composite weight for {module.__class__.__name__} by concatenating {len(weight_params)} parameters")
                    return composite
            
            # Strategy 2: For MLP-like modules with incompatible dimensions
            # Check if this looks like an MLP (W_in and W_out pattern)
            if len(weight_params) == 2:
                names = [name for name, _ in weight_params]
                if any('in' in n.lower() for n in names) and any('out' in n.lower() for n in names):
                    # This is likely an MLP - use the input weight as representative
                    for name, param in weight_params:
                        if 'in' in name.lower():
                            logger.debug(f"MLP-like module {module.__class__.__name__}: using input weight as representative")
                            if param.dim() == 2:
                                return param
                            else:
                                return param.view(-1, param.size(-1))
            
            # Strategy 3: Use the largest weight as representative
            sorted_params = sorted(weight_params, key=lambda x: x[1].numel(), reverse=True)
            largest_name, largest_param = sorted_params[0]
            
            if len(weight_params) > 1:
                logger.debug(f"Module {module.__class__.__name__} has incompatible weights {[p[0] for p in weight_params]}, using {largest_name} as representative")
            
            if largest_param.dim() == 2:
                return largest_param
            else:
                return largest_param.view(-1, largest_param.size(-1))
            
        except Exception as e:
            logger.warning(f"Could not create composite weight for {module.__class__.__name__}: {e}")
        
        return None
    
    def _create_composite_bias(self, module, bias_params):
        """Create a composite bias tensor from multiple bias parameters."""
        try:
            # Concatenate all bias parameters
            biases = [param for _, param in bias_params]
            if biases:
                composite = torch.cat(biases, dim=0)
                return composite
        except Exception as e:
            logger.warning(f"Could not create composite bias for {module.__class__.__name__}: {e}")
        
        return None
    
    def forward(self, *args, **kwargs):
        """Forward pass through original model."""
        return self.original_model(*args, **kwargs)
    
    def modules(self):
        """Return modules from the original model with weight attributes attached."""
        # This ensures metrics can find the weight attributes we created
        return self.original_model.modules()
    
    def named_modules(self, *args, **kwargs):
        """Return named modules from the original model."""
        return self.original_model.named_modules(*args, **kwargs)
    
    def children(self):
        """Return children from the original model."""
        return self.original_model.children()
    
    def named_children(self):
        """Return named children from the original model."""
        return self.original_model.named_children()
    
    def parameters(self, recurse=True):
        """Return parameters from the original model."""
        return self.original_model.parameters(recurse=recurse)
    
    def named_parameters(self, *args, **kwargs):
        """Return named parameters from the original model."""
        return self.original_model.named_parameters(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_model, name)
    
    def state_dict(self, *args, **kwargs):
        """Return the original model's state dict."""
        return self.original_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the original model."""
        return self.original_model.load_state_dict(*args, **kwargs)


def wrap_model_for_metrics(model):
    """
    Wrap a model to ensure compatibility with standard metrics.
    
    This function checks if the model has non-standard parameter names or 3D+ tensors
    and wraps it in MetricsCompatibleModel if needed.
    """
    needs_wrapping = False
    
    # Check if model has non-standard parameters
    for name, module in model.named_modules():
        if module is model:
            continue
        
        # Get all parameters in this module
        module_params = list(module.named_parameters(recurse=False))
        if not module_params:
            continue
        
        # Check for non-standard weight parameters
        has_standard_weight = hasattr(module, 'weight') and module.weight is not None
        weight_like_params = []
        
        for param_name, param in module_params:
            if param is not None and param.dim() >= 2:
                if any(indicator in param_name.upper() for indicator in ['W', 'WEIGHT', 'KERNEL']):
                    weight_like_params.append(param_name)
                elif param_name in ['embedding', 'embed']:
                    weight_like_params.append(param_name)
        
        # Need wrapping if we have weight-like params but no standard weight
        if weight_like_params and not has_standard_weight:
            needs_wrapping = True
            logger.debug(f"Module {module.__class__.__name__} has non-standard weight parameters: {weight_like_params}")
            break
        
        # Also check for 3D+ parameters
        for param_name, param in module_params:
            if param is not None and param.dim() > 2:
                needs_wrapping = True
                logger.debug(f"Module {module.__class__.__name__} has {param.dim()}D parameter: {param_name}")
                break
        
        if needs_wrapping:
            break
    
    if needs_wrapping:
        logger.info("Model has non-standard parameters, wrapping for metrics compatibility")
        return MetricsCompatibleModel(model)
    
    return model


def get_all_weight_parameters(model, include_all=False):
    """
    Get all weight parameters from a model in a standardized way.
    
    Args:
        model: The model to extract weights from
        include_all: If True, include all parameters. If False, only 2D+ weight-like parameters.
        
    Returns:
        List of (module_name, param_name, parameter) tuples
    """
    weights = []
    
    # Handle wrapped models
    if isinstance(model, MetricsCompatibleModel):
        model = model.original_model
    
    for module_name, module in model.named_modules():
        # First try standard weight attribute
        if hasattr(module, 'weight') and module.weight is not None:
            weights.append((module_name, 'weight', module.weight))
        else:
            # Look for non-standard weight parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param is not None:
                    if include_all or (
                        param.dim() >= 2 and 
                        any(ind in param_name.upper() for ind in ['W', 'WEIGHT', 'KERNEL', 'EMBED'])
                    ):
                        weights.append((module_name, param_name, param))
    
    return weights


def get_weight_matrices_for_metrics(model):
    """
    Get all weight matrices from a model, properly handling wrapped models.
    This is specifically for metrics that expect 2D weight matrices.
    
    Returns:
        List of 2D tensors
    """
    matrices = []
    
    # Make sure we're looking at the right model
    target_model = model.original_model if isinstance(model, MetricsCompatibleModel) else model
    
    for name, module in target_model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            w = module.weight
            if w.dim() == 2 and w.numel() > 0:
                matrices.append(w)
            elif w.dim() > 2:
                # Flatten higher dimensional tensors
                flattened = w.view(-1, w.size(-1))
                if flattened.numel() > 0:
                    matrices.append(flattened)
    
    # If no matrices found through weight attribute, look for parameters directly
    if not matrices:
        for name, param in target_model.named_parameters():
            if param.dim() >= 2 and any(ind in name.upper() for ind in ['W', 'WEIGHT', 'KERNEL', 'EMBED']):
                if param.dim() == 2:
                    matrices.append(param)
                else:
                    flattened = param.view(-1, param.size(-1))
                    matrices.append(flattened)
    
    return matrices
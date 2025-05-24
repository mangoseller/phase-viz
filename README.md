# Phase-Viz 

**Phase-Viz** is a plug-and-play visualization tool for analyzing the developmental trajectory of neural networks during training. It aims to helps researchers understand how model geometry evolves across training checkpoints and detect transitions in network behavior.

## Overview

Phase-Viz provides an intuitive way to:
- Track multiple metrics across training checkpoints
- Detect phase transitions in model development
- Visualize training dynamics with interactive plots
- Support custom metrics and model architectures

The tool creates temporary interactive web visualizations that automatically clean up after use, with all processing logged to rotating log files.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- `torch>=2.7.0`
- `typer>=0.15.4`
- `matplotlib>=3.10.3`

## Quick Start

1. **Load your model checkpoints:**
```bash
python src/cli.py load-dir --dir path/to/checkpoints --model model.py --class-name ModelClass
```

2. **Visualize metrics:**
```bash
python src/cli.py plot-metric
```

When prompted, enter metric names (e.g., `l2`, `entropy`) or provide a custom metrics file. Type `done` when finished selecting metrics.

## Commands

### `load-dir`
Loads a directory containing model checkpoints (`.pt` or `.ckpt` files).

**Options:**
- `--dir`: Directory containing model checkpoints
- `--model`: Path to Python file defining the model class
- `--class-name`: Name of the model class inside the file

**Example:**
```bash
python src/cli.py load-dir --dir ./checkpoints --model models/resnet.py --class-name ResNet18
```

### `plot-metric`
Computes and visualizes selected metrics across all loaded checkpoints.

**Options:**
- `--device`: Device to use for calculations (`cuda`, `cpu`, or specific `cuda:n`) [default: cuda]
- `--parallel`: Use parallel processing for calculations [default: True]

**Interactive Commands:**
- `metrics`: List all available built-in metrics
- `l2`: Add L2 norm metric
- `entropy`: Add weight entropy metric
- `connectivity`: Add layer connectivity metric
- `path/to/file.py`: Load custom metrics from a Python file
- `done`: Finish selection and start visualization
- `exit`: Exit without processing

## Built-in Metrics

Phase-Viz includes several pre-built metrics for analyzing neural network geometry:

### L2 Norm
**Command:** `l2`

Computes the L2 norm of all trainable parameters:

$$\|w\|_2 = \sqrt{\sum_{i} w_i^2}$$

This metric tracks the overall magnitude of model weights, useful for monitoring weight growth or decay during training.

### Weight Entropy
**Command:** `entropy`

Calculates the Shannon entropy of the weight distribution:

$$H(W) = -\sum_{i=1}^{n} p_i \log(p_i)$$

where $p_i$ is the probability of weights falling in bin $i$. Higher entropy indicates more uniform weight distribution, while lower entropy suggests weights concentrated around specific values.

### Layer Connectivity
**Command:** `connectivity`

Measures the average absolute weight per layer:

$$C = \frac{1}{L} \sum_{l=1}^{L} \frac{1}{N_l} \sum_{i,j} |w_{ij}^{(l)}|$$

where $L$ is the number of layers and $N_l$ is the number of parameters in layer $l$. This metric indicates the average strength of connections in the network.

## Custom Metrics

You can define custom metrics in a Python file. Each metric function must:
1. End with `_of_model`
2. Take exactly one argument (the model)
3. Return a float

**Example custom metric:**
```python
def sparsity_of_model(model: torch.nn.Module) -> float:
    """Calculate the fraction of near-zero weights."""
    threshold = 1e-3
    total = 0
    near_zero = 0
    
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
            near_zero += (p.abs() < threshold).sum().item()
    
    return near_zero / total if total > 0 else 0.0
```

Load custom metrics by providing the file path when prompted:
```
Enter a metric name: custom_metrics.py
```

## Features

### Interactive Visualization
- Click charts to expand for detailed view
- Toggle between individual and overlay views for multiple metrics
- Export data as CSV or download plots as PNG

### Phase Transition Detection
The tool automatically identifies significant changes in metrics (>20% relative change between checkpoints) and highlights these transitions in the visualization.


### Logging System
All operations are logged to `logs/phase-viz.log` with automatic rotation after 10,000 lines. The logging system captures:
- Model loading events
- Metric computation progress
- Errors and warnings
- Performance metrics

### Temporary Web Interface
Phase-Viz creates a temporary HTML file for visualization that:
- Opens automatically in your default browser
- Includes a cleanup server that removes the file when closed


## Example Workflow

```bash

# 1. Load the checkpoints
python src/cli.py load-dir \
    --dir large_checkpoints \
    --model largetest.py \
    --class-name LargeNet

# 2. Visualize metrics
python src/cli.py plot-metric --device cuda

# When prompted:
> metrics          # See available metrics
> l2              # Add L2 norm
> entropy         # Add weight entropy
> done            # Start visualization
> exit            # Exit the tool
```

## Architecture Support

Phase-Viz automatically detects model architecture from checkpoints and supports:
- Standard PyTorch models
- Models with custom initialization
- Transformer architectures
- Convolutional networks
- Recurrent networks (LSTM, GRU)

The tool intelligently extracts configuration parameters from state dictionaries, eliminating the need for manual configuration files.


### Metric Computation Errors
Failed metrics return `NaN` values without stopping the entire process. Check the log file for detailed error messages.


## License

This project is open source and available under the MIT License.
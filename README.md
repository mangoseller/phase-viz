# Phase-Viz 

![image](https://github.com/user-attachments/assets/0cf14db2-8de0-481c-9996-6edccbc641db)


Phase-Viz is a visualization tool for generically analyzing the developmental trajectory of neural networks during training. It aims to assist in understanding how model weight geometry evolves throughout the learning process and help link quantitative changes in model statistics with qualitative changes in model behaviour.

## Overview

**Phase-Viz** provides an intuitive way to:
- Track multiple metrics across training checkpoints
- Detect [phase transitions](https://www.lesswrong.com/posts/aKBAYN5LpaQMrPqMj/dslt-4-phase-transitions-in-neural-networks) during training
- Visualize training dynamics with interactive plots
 
Phase-viz supports custom metrics and model architectures, and creates temporary interactive web visualizations that are automatically cleaned up after use, with all processing logged to rotating log files.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- `numpy==2.2.6` 
- `pytest==7.4.4`
- `scipy==1.15.3`
- `torch==2.7.0`
- `typer==0.16.0`


## Quick Start

1. **Load model checkpoints:**
```bash
python src/cli.py load-dir --dir path/to/checkpoints --model model.py --class-name ModelClass
```

2. **Visualize metrics:**
```bash
python src/cli.py plot-metrics
```

When prompted, enter metric names (e.g., `l2`, `entropy`) or provide a custom metrics file. Type `done` when finished selecting metrics.

## Commands

### `load-dir`
Loads a directory containing model checkpoints (`.pt` or `.ckpt` files).

**Options:**
- `--dir`: Directory containing model checkpoints
- `--model`: Path to Python file defining the model class
- `--class-name`: Name of the model class inside the file (If not provided - `class-name` will attempt to be inferred from the file provided to `--model`)

**Example:**
```bash
python cli.py load-dir --dir ../grok_1748461191 --model ../examples/transformer_grokking_example.py --class-name GrokkingMLP
```

### `plot-metrics`
Computes and visualizes selected metrics across all loaded checkpoints.

**Options:**
- `--device`: Device to use for calculations (`cuda`, `cpu`, or specific `cuda:n`) [default: cuda]
- `--parallel`: Use parallel processing for calculations [default: True]

**Interactive Commands:**
- `metrics`: List all available built-in metrics
- `path/to/file.py`: Load custom metrics from a Python file
- `done`: Finish selection and start visualization
- `exit`: Exit without processing

## Built-in Metrics

Phase-Viz includes several pre-built metrics:

### Weight Entropy
**Command:** `entropy`

Calculates the Shannon entropy of the weight distribution:

$$H(W) = -\sum_{i=1}^{n} p_i \log(p_i)$$

where $p_i$ is the probability of weights falling in bin $i$. Higher entropy indicates more uniform weight distribution, while lower entropy suggests weights concentrated around specific values.

### Layer Connectivity
**Command:** `connectivity`

Measures the average absolute weight per layer:

$$C = \frac{1}{L} \sum_{l=1}^{L} \text{mean}(|W^{(l)}|)$$

where $L$ is the number of layers and $W^{(l)}$ represents the weights in layer $l$. This metric indicates the average strength of connections in the network.

### Parameter Variance
**Command:** `variance`

Computes the variance of all trainable parameters:

$$\text{Var}(W) = \frac{1}{N} \sum_{i=1}^{N} (w_i - \bar{w})^2$$

where $\bar{w}$ is the mean weight. This metric helps track how spread out the parameter values are.

### Layer Wise Norm Ratio
**Command:** `norm_ratio`

Computes the ratio of norms between first and last layers:

![equation](https://latex.codecogs.com/svg.image?\dpi{200}\color{White}R%20=%20\frac{\|W_{\text{last}}\|_2}{\|W_{\text{first}}\|_2})




This can help identify gradient vanishing/exploding issues.

### Activation Capacity
**Command:** `capacity`

Estimates the model's activation capacity based on layer dimensions:

$$C = \sum_{\text{layers}} \log(d_{\text{out}} + 1)$$

where $d_{\text{out}}$ is the output dimension of each layer. This is a proxy for the model's representational capacity.

### Dead Neuron Percentage
**Command:** `dead_neurons`

Estimates percentage of "dead" neurons (near-zero weights):

$$D = \frac{|\{w : |w| < \epsilon\}|}{|W|} \times 100\%$$

where $\epsilon = 10^{-6}$. This can indicate over-regularization or training issues.

### Weight Rank
**Command:** `rank`

Computes average effective rank of weight matrices:

$$\text{EffRank}(W) = \exp\left(-\sum_{i} \frac{s_i}{\sum_j s_j} \log\left(\frac{s_i}{\sum_j s_j}\right)\right)$$

where $s_i$ are the singular values. Lower rank might indicate redundancy in the model.

### Gradient Flow Score
**Command:** `gradient_flow`

Computes a score representing potential gradient flow quality:

$$\text{GFS} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{1 + |\sigma_i - \sigma_{\text{expected}}|}$$

where $\sigma_i$ is the standard deviation of layer $i$ and $\sigma_{\text{expected}}$ is based on Xavier/He initialization.

### Effective Rank
**Command:** `effective_rank`

Computes the average effective rank of all weight matrices using entropy of singular values:

$$\text{EffRank}(W) = \exp(H(s))$$

where $H(s)$ is the entropy of normalized singular values.

### Average Condition Number
**Command:** `condition`

Computes the average condition number of weight matrices:

$$\kappa = \frac{\sigma_{\max}}{\sigma_{\min}}$$

### Flatness Proxy
**Command:** `flatness`

Computes a proxy for loss landscape flatness:

$$F = \|W\|_F \times \|W\|_2$$

where $\|W\|_F$ is the Frobenius norm and $\|W\|_2$ is the spectral norm.

### Mean Weight
**Command:** `mean`

Computes the mean of all trainable weights:

$$\bar{w} = \frac{1}{N} \sum_{i=1}^{N} w_i$$

Useful for detecting weight drift or bias.

### Weight Skew
**Command:** `skew`

Computes the skewness of weight distribution:

$$\text{Skew}(W) = \frac{\mathbb{E}\left[(W - \mu)^3\right]}{\sigma^3}$$

Measures the asymmetry of weight distribution.

### Weight Kurtosis
**Command:** `kurtosis`

Computes the kurtosis of weight distribution:

$$\text{Kurt}(W) = \frac{\mathbb{E}\left[(W - \mu)^4\right]}{\sigma^4}$$

Measures the "tailedness" of weight distribution.

### Isotropy
**Command:** `isotropy`

Computes the isotropy of weight matrices:

$$I = \frac{\text{tr}(W W^\top)^2}{\|W W^\top\|_F^2}$$

Values closer to 1 indicate more isotropic (uniform) distributions.

### Weight Norm
**Command:** `weight_norm`

Computes the Frobenius norm of all trainable parameters:

![Frobenius norm](https://latex.codecogs.com/svg.image?\dpi{200}\color{White}|W|_F%20=%20\sqrt{\sum_{i,j}%20w_{ij}^2})


### Spectral Norm
**Command:** `spectral`

Computes the maximum singular value across all weight matrices:

![Spectral norm](https://latex.codecogs.com/svg.image?\dpi{200}\color{White}|W|_2%20=%20\sigma_{\max}(W))


### Participation Ratio
**Command:** `participation`

Computes the participation ratio of all trainable parameters:

$$PR = \frac{\left(\sum_i w_i^2\right)^2}{\sum_i w_i^4}$$

Higher values indicate more even distribution of weights.

### Sparsity
**Command:** `sparsity`

Computes the fraction of near-zero parameters:

$$S = \frac{|\{w : |w| < \epsilon\}|}{|W|}$$

where $\epsilon = 10^{-3}$.

### Max Activation
**Command:** `max_activation`

Estimates the maximum potential activation:

![activation capacity](https://latex.codecogs.com/svg.image?\dpi{200}\color{White}\mathcal{C}%20=%20\frac{1}{L}%20\sum_{l=1}^{L}%20\log(m_l%20\cdot%20n_l)%20\cdot%20\frac{\exp\left(-\sum_{i=1}^{r_l}%20p_i^{(l)}%20\log(p_i^{(l)}%20+%2010^{-10})\right)}{r_l})


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
Phase-viz automatically identifies significant changes in metrics (>20% relative change between checkpoints) and highlights these transitions in the visualization.


### Logging System
All operations are logged to `logs/phase-viz.log` with automatic rotation after 10,000 lines. The logging system captures:
- Model loading events
- Metric computation progress
- Errors and warnings
- Performance metrics

### Temporary Web Interface
Phase-Viz creates a temporary HTML file for visualization that:
- Opens automatically in your default browser
- Includes a cleanup server that removes the file when the page is closed


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
> spectral        # Add Spectral Norm
> entropy         # Add weight entropy
> done            # Start visualization
> exit            # Exit the tool
```

## Architecture Support

Phase-viz attempts to automatically detect model architecture from checkpoints and supports:
- Standard PyTorch models
- Models with custom initialization
- Transformer architectures
- Convolutional networks
- Recurrent networks

## Examples
- Example models and checkpoints are stored in the `examples` directory.

### Grokking Transformer

- This is the transformer architecture featured in the paper [Progress measures for grokking via mechanistic interpretability](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/tree/main) 
- Note: `transformer_checkpoints_example` contains over 50 checkpoints, and can take 6-7 minutes to compute and render multiple metrics. Consider using CUDA if you have access to a GPU. 

```bash
# Load the checkpoints
python src/cli.py load-dir \
--dir ../examples/transformer_checkpoints_example/ \
--model ../examples/transformer_grokking_example.py \
--class-name Transformer \ 

# Select metrics
python src/cli.py plot-metric --device cuda
```

![image](https://github.com/user-attachments/assets/79324a54-9c72-4f04-be60-2e837e731d9f)
*Weight entropy over training - checkpointed every 1000 epochs*
### Deep Linear Network

- Deep linear architecture as featured in [Timaeus's dev interp examples repo](https://github.com/timaeus-research/devinterp/blob/main/examples/dlns.ipynb)

```bash
# Load the checkpoints
python src/cli.py load-dir 
--dir ../examples/DLN_checkpoints_example/ \
--model ../examples/DLN_example.py \
--class-name DLN \

# Select metrics
python src/cli.py plot-metric --device cuda
```

![image](https://github.com/user-attachments/assets/272e31e0-2444-4e4b-a3f1-74d11c04ddff)

*Overlaying and normalizing two metrics - Gradient Flow and Spectral Norm*

## License

This project is open source and available under the MIT License.

"""Generate test checkpoints for various model architectures."""

import os
import torch
import torch.nn as nn
from pathlib import Path
import time

# Import test models
from test_models import (
    SimpleNet, ConfigurableNet, TransformerModel, ConvNet, 
    RNNModel, CustomConfigModel, NoParamsModel, DynamicModel
)


def create_checkpoint(model, checkpoint_path, epoch=1, include_config=False, 
                     config=None, include_optimizer=False):
    """Create a checkpoint with various formats."""
    checkpoint = {}
    
    # Different checkpoint formats to test
    if include_config and config:
        checkpoint['config'] = config
        checkpoint['model_state'] = model.state_dict()
    elif include_optimizer:
        # Simulate training checkpoint with optimizer
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint['epoch'] = epoch
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['loss'] = 0.5 - 0.1 * epoch
    else:
        # Simple format - just state dict
        checkpoint = model.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  Created: {checkpoint_path}")


def generate_checkpoints_for_model(model_name, model_class, model_kwargs=None, 
                                  num_checkpoints=3, output_dir="test_checkpoints"):
    """Generate multiple checkpoints for a given model."""
    print(f"\nGenerating checkpoints for {model_name}...")
    
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model instance
    if model_kwargs:
        model = model_class(**model_kwargs)
    else:
        model = model_class()
    
    # Generate checkpoints with different formats
    for i in range(num_checkpoints):
        # Reinitialize weights
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU, nn.RNN)):
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Create different checkpoint formats
        if i == 0:
            # Simple state dict
            checkpoint_path = model_dir / f"checkpoint_{i+1:03d}.pt"
            create_checkpoint(model, checkpoint_path)
        elif i == 1:
            # With config
            checkpoint_path = model_dir / f"checkpoint_{i+1:03d}.pt"
            config = model_kwargs if model_kwargs else {}
            create_checkpoint(model, checkpoint_path, epoch=i+1, 
                            include_config=True, config=config)
        else:
            # With optimizer
            checkpoint_path = model_dir / f"checkpoint_{i+1:03d}.pt"
            create_checkpoint(model, checkpoint_path, epoch=i+1, 
                            include_optimizer=True)
        
        time.sleep(0.1)  # Ensure different timestamps


def generate_edge_case_checkpoints(output_dir="test_checkpoints/edge_cases"):
    """Generate checkpoints for edge cases."""
    print("\nGenerating edge case checkpoints...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Empty checkpoint
    torch.save({}, Path(output_dir) / "empty.pt")
    print(f"  Created: {output_dir}/empty.pt")
    
    # Corrupted checkpoint (invalid tensor)
    torch.save({"model_state": "not_a_tensor"}, Path(output_dir) / "corrupted.pt")
    print(f"  Created: {output_dir}/corrupted.pt")
    
    # Checkpoint with extra keys
    model = SimpleNet()
    checkpoint = {
        "model_state": model.state_dict(),
        "extra_key": "extra_value",
        "metadata": {"version": "1.0"}
    }
    torch.save(checkpoint, Path(output_dir) / "extra_keys.pt")
    print(f"  Created: {output_dir}/extra_keys.pt")


def main():
    """Generate all test checkpoints."""
    print("=== Generating Test Checkpoints for phase-viz ===")
    
    # Clean up old checkpoints
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")
    
    # Generate checkpoints for each model type
    models_to_test = [
        ("simple_net", SimpleNet, None),
        ("configurable_net", ConfigurableNet, {
            "input_dim": 20, 
            "hidden_size": 128, 
            "num_layers": 4, 
            "output_dim": 2
        }),
        ("transformer", TransformerModel, {
            "vocab_size": 500,
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 1024
        }),
        ("conv_net", ConvNet, {"num_classes": 10}),
        ("rnn_lstm", RNNModel, {
            "input_size": 50,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 5,
            "rnn_type": "LSTM"
        }),
        ("rnn_gru", RNNModel, {
            "input_size": 50,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 5,
            "rnn_type": "GRU"
        }),
        ("no_params", NoParamsModel, None),
        ("dynamic_model", DynamicModel, {"base_dim": 32}),
    ]
    
    for model_name, model_class, kwargs in models_to_test:
        generate_checkpoints_for_model(model_name, model_class, kwargs)
    
    # Generate edge case checkpoints
    generate_edge_case_checkpoints()
    
    # Special case: CustomConfigModel
    print("\nGenerating checkpoints for CustomConfigModel...")
    model_dir = Path("test_checkpoints/custom_config")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "input_dim": 15,
        "hidden_dim": 64,
        "output_dim": 3,
        "use_dropout": True,
        "dropout_rate": 0.3
    }
    model = CustomConfigModel(config)
    checkpoint = {
        "config": config,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, model_dir / "checkpoint_001.pt")
    print(f"  Created: {model_dir}/checkpoint_001.pt")
    
    print("\n✅ All test checkpoints generated successfully!")
    print("\nYou can now run the tests with: pytest test_phase_viz.py -v")


if __name__ == "__main__":
    main()
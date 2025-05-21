import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time

# Import the simplified LargeNet class
from largetest import LargeNet

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_model_random(model):
    """Reinitialize all model parameters randomly"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
def create_checkpoints(num_checkpoints=3, output_dir="checkpoints", hidden_size=768, num_blocks=6):
    """Create multiple checkpoints of the model with different random initializations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the model with specified parameters
    model = LargeNet(hidden_size=hidden_size, num_blocks=num_blocks)
    
    # Count parameters and print model size
    num_params = count_parameters(model)
    print(f"Model created with {num_params:,} trainable parameters")
    print(f"Model configuration: hidden_size={hidden_size}, num_blocks={num_blocks}")
    
    # Generate checkpoints
    for i in range(num_checkpoints):
        print(f"\nGenerating checkpoint {i+1}/{num_checkpoints}...")
        
        # Reinitialize model weights randomly
        initialize_model_random(model)
        
        # Create a fake optimizer state to simulate training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create some fake metrics to add to the checkpoint
        fake_metrics = {
            "loss": 0.5 - 0.1 * i,
            "accuracy": 0.7 + 0.05 * i,
            "epoch": i + 1
        }
        
        # Create the checkpoint - simple format with just model state dict
        checkpoint = model.state_dict()
        
        # Save the checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1:03d}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Add a small delay to ensure different timestamps
        time.sleep(0.5)
        
    print(f"\nSuccessfully created {num_checkpoints} checkpoints in {output_dir}/")
    print(f"You can use these checkpoints with the phase-viz tool:")
    print(f"  python src/cli.py load-dir --dir {output_dir} --model large_model_simplified.py --class-name LargeNet")

if __name__ == "__main__":
    create_checkpoints(
        num_checkpoints=10,  # Generate 10 checkpoints to test loading
        output_dir="large_checkpoints",
        hidden_size=768,
        num_blocks=3
    )

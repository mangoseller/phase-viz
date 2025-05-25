import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock



# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))
# Import modules to test
from load_models import (
    load_model_class, extract_model_config_from_state_dict,
    initialize_model_with_config, load_model_from_checkpoint,
    contains_checkpoints, clear_model_cache
)
from inbuilt_metrics import l2_norm_of_model
from metrics import (
    compute_metric_batch, 
    compute_metrics_over_checkpoints, import_metric_functions
)
from state import save_state, load_state
from utils import logger

# Import test models
from test_models import (
    SimpleNet, ConfigurableNet, TransformerModel, ConvNet,
    RNNModel, CustomConfigModel, NoParamsModel, DynamicModel
)


class TestModelLoading:
    """Test model loading functionality."""
    
    @pytest.fixture
    def setup_test_env(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.model_file = Path(self.test_dir) / "test_models.py"
        
        # Copy test models to temp directory
        shutil.copy("test_models.py", self.model_file)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir)
        clear_model_cache()
    
    def test_load_model_class(self, setup_test_env):
        """Test loading model classes from file."""
        # Test loading SimpleNet
        model_class = load_model_class(str(self.model_file), "SimpleNet")
        assert model_class.__name__ == "SimpleNet"
        
        # Test model instantiation
        model = model_class()
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_load_missing_model_class(self, setup_test_env):
        """Test error handling for missing model class."""
        with pytest.raises(ValueError, match="Class 'NonExistentModel' not found"):
            load_model_class(str(self.model_file), "NonExistentModel")
    
    def test_extract_config_from_state_dict(self):
        """Test configuration extraction from state dict."""
        # Test with ConfigurableNet
        model = ConfigurableNet(input_dim=20, hidden_size=128, num_layers=4)
        state_dict = model.state_dict()
        
        # The extraction function should handle various architectures
        config = extract_model_config_from_state_dict(state_dict)
        assert isinstance(config, dict)
        assert 'hidden_size' in config
        
        # Test with TransformerModel
        transformer = TransformerModel(d_model=256, nhead=4)
        state_dict = transformer.state_dict()
        config = extract_model_config_from_state_dict(state_dict)
        assert isinstance(config, dict)
    
    def test_initialize_model_with_config(self):
        """Test model initialization with different config approaches."""
        # Test with explicit parameters
        config = {"input_dim": 15, "hidden_size": 64, "num_layers": 3}
        model = initialize_model_with_config(ConfigurableNet, config)
        assert isinstance(model, ConfigurableNet)
        
        # Test with config parameter
        config = {"input_dim": 10, "hidden_dim": 32, "output_dim": 2}
        model = initialize_model_with_config(CustomConfigModel, config)
        assert isinstance(model, CustomConfigModel)
        
        # Test with default initialization
        model = initialize_model_with_config(SimpleNet, {})
        assert isinstance(model, SimpleNet)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_load_model_from_checkpoint(self, device):
        """Test loading models from checkpoints."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model = SimpleNet()
            torch.save(model.state_dict(), tmp.name)
            
            # Load model class first
            clear_model_cache()
            with patch('load_models._model_class', SimpleNet):
                loaded_model = load_model_from_checkpoint(tmp.name, device)
                assert isinstance(loaded_model, SimpleNet)
                assert next(loaded_model.parameters()).device.type == device
            
            os.unlink(tmp.name)
    
    def test_load_checkpoint_formats(self):
        """Test loading different checkpoint formats."""
        test_cases = [
            # Format 1: Just state dict
            lambda m: m.state_dict(),
            # Format 2: Dict with model_state
            lambda m: {"model_state": m.state_dict()},
            # Format 3: Dict with state_dict
            lambda m: {"state_dict": m.state_dict()},
            # Format 4: Full training checkpoint
            lambda m: {
                "epoch": 10,
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": {},
                "loss": 0.5
            }
        ]
        
        for i, create_checkpoint in enumerate(test_cases):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                model = SimpleNet()
                checkpoint = create_checkpoint(model)
                torch.save(checkpoint, tmp.name)
                
                with patch('load_models._model_class', SimpleNet):
                    try:
                        loaded_model = load_model_from_checkpoint(tmp.name)
                        assert isinstance(loaded_model, SimpleNet)
                    except Exception as e:
                        pytest.fail(f"Failed to load checkpoint format {i}: {e}")
                
                os.unlink(tmp.name)

class TestMetrics:
    """Test metric computation functionality."""
    
    def test_l2_norm_computation(self):
        """Test L2 norm metric computation."""
        model = SimpleNet()
        norm = l2_norm_of_model(model)
        assert isinstance(norm, float)
        assert norm > 0
        
        # Test with model with no parameters
        no_param_model = NoParamsModel()
        norm = l2_norm_of_model(no_param_model)
        assert norm == 0.0
    
    def test_compute_metric_batch(self):
        """Test batch metric computation."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model = SimpleNet()
            torch.save(model.state_dict(), tmp.name)
            
            metrics = {
                "L2 Norm": l2_norm_of_model,
                "Custom": lambda m: 42.0
            }
            
            with patch('load_models._model_class', SimpleNet):
                results = compute_metric_batch(metrics, tmp.name)
                assert "L2 Norm" in results
                assert "Custom" in results
                assert results["Custom"] == 42.0
            
            os.unlink(tmp.name)
    
    def test_metric_error_handling(self):
        """Test metric computation with errors."""
        def failing_metric(model):
            raise ValueError("Metric computation failed")
        
        def none_metric(model):
            return None
        
        def string_metric(model):
            return "not a number"
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model = SimpleNet()
            torch.save(model.state_dict(), tmp.name)
            
            metrics = {
                "Failing": failing_metric,
                "None": none_metric,
                "String": string_metric
            }
            
            with patch('load_models._model_class', SimpleNet):
                results = compute_metric_batch(metrics, tmp.name)
                assert np.isnan(results["Failing"])
                assert np.isnan(results["None"])
                assert isinstance(results["String"], float)  # Should convert to float
            
            os.unlink(tmp.name)
    
    @pytest.mark.parametrize("parallel,device", [
        (True, "cpu"),
        (False, "cpu"),
        (True, "cuda"),
        (False, "cuda")
    ])
    def test_compute_metrics_over_checkpoints(self, parallel, device):
        """Test computing metrics over multiple checkpoints."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create temporary checkpoints
        checkpoint_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                model = SimpleNet()
                torch.save(model.state_dict(), tmp.name)
                checkpoint_files.append(tmp.name)
        
        try:
            metrics = {"L2 Norm": l2_norm_of_model}
            
            with patch('load_models._model_class', SimpleNet):
                results = compute_metrics_over_checkpoints(
                    metrics, checkpoint_files, device=device, parallel=parallel
                )
                print(metrics)
                assert "L2 Norm" in results
                assert len(results["L2 Norm"]) == 3
                assert all(isinstance(v, float) for v in results["L2 Norm"])
        finally:
            for f in checkpoint_files:
                os.unlink(f)
    
    def test_import_metric_functions(self):
        """Test importing custom metric functions."""
        # Create a temporary metric file
        metric_code = '''
import torch

def custom_metric_of_model(model):
    """Custom metric function."""
    return 123.456

def another_metric_of_model(model):
    """Another metric."""
    total = 0.0
    for p in model.parameters():
        total += p.numel()
    return float(total)

def not_a_metric(model):
    """This should not be imported (wrong name)."""
    return 0.0

def wrong_args_of_model(model, extra_arg):
    """This should not be imported (wrong args)."""
    return 0.0
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(metric_code)
            tmp.flush()
            
            metrics = import_metric_functions(tmp.name)
            
            assert "Custom Metric" in metrics
            assert "Another Metric" in metrics
            assert "not_a_metric" not in metrics
            assert "wrong_args" not in metrics
            
            # Test the imported functions
            model = SimpleNet()
            assert metrics["Custom Metric"](model) == 123.456
            assert isinstance(metrics["Another Metric"](model), float)
            
            os.unlink(tmp.name)


class TestTransformerArchitecture:
    """Specific tests for transformer architecture loading."""
    
    def test_transformer_loading(self):
        """Test loading and computing metrics on transformer model."""
        # Create transformer checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model = TransformerModel(
                vocab_size=1000,
                d_model=256,
                nhead=4,
                num_layers=2
            )
            checkpoint = {
                "config": {
                    "vocab_size": 1000,
                    "d_model": 256,
                    "nhead": 4,
                    "num_layers": 2
                },
                "state_dict": model.state_dict()
            }
            torch.save(checkpoint, tmp.name)
            
            with patch('load_models._model_class', TransformerModel):
                loaded_model = load_model_from_checkpoint(tmp.name)
                assert isinstance(loaded_model, TransformerModel)
                
                # Compute metrics
                norm = l2_norm_of_model(loaded_model)
                assert isinstance(norm, float)
                assert norm > 0
            
            os.unlink(tmp.name)
    
    def test_transformer_different_configs(self):
        """Test transformer with various configurations."""
        configs = [
            {"vocab_size": 500, "d_model": 128, "nhead": 4, "num_layers": 2},
            {"vocab_size": 2000, "d_model": 512, "nhead": 8, "num_layers": 6},
            {"vocab_size": 100, "d_model": 64, "nhead": 2, "num_layers": 1},
        ]
        
        for config in configs:
            model = TransformerModel(**config)
            state_dict = model.state_dict()
            
            # Test configuration extraction
            extracted_config = extract_model_config_from_state_dict(state_dict)
            assert isinstance(extracted_config, dict)
            
            # Ensure model can be recreated
            new_model = TransformerModel(**config)
            new_model.load_state_dict(state_dict)


class TestStateManagement:
    """Test state persistence functionality."""
    
    def test_save_and_load_state(self):
        """Test saving and loading state."""
        test_state = {
            "dir": "/path/to/checkpoints",
            "model_path": "model.py",
            "class_name": "TestModel",
            "checkpoints": ["checkpoint1.pt", "checkpoint2.pt"],
            "metrics_data": {"L2 Norm": [1.0, 2.0]}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            save_state(test_state)
            loaded_state = load_state()
            
            assert loaded_state == test_state
    
    def test_load_missing_state(self):
        """Test error handling for missing state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            with pytest.raises(RuntimeError, match="No state found"):
                load_state()

class TestIntegration:
    """Integration tests for the full workflow."""    
    def test_end_to_end_workflow(self):
        """Test complete workflow from loading to metric computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model file
            model_file = Path(tmpdir) / "models.py"
            test_models_path = Path(__file__).parent / "test_models.py"
            shutil.copy(test_models_path, model_file)
            
            # Create checkpoints directory
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            
            # Generate checkpoints
            for i in range(3):
                model = SimpleNet()
                checkpoint_path = checkpoint_dir / f"checkpoint_{i:03d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
            
            # Test checkpoint discovery
            checkpoints = contains_checkpoints(str(checkpoint_dir))
            assert len(checkpoints) == 3
            
            # Load model class
            load_model_class(str(model_file), "SimpleNet")
            
            # Compute metrics
            metrics = {"L2 Norm": l2_norm_of_model}
            results = compute_metrics_over_checkpoints(
                metrics, checkpoints, device="cpu", parallel=False
            )
            
            assert "L2 Norm" in results
            assert len(results["L2 Norm"]) == 3
            assert all(isinstance(v, float) for v in results["L2 Norm"])
    
    def test_various_architectures(self):
        """Test loading and computing metrics on various architectures."""
        architectures = [
            ("SimpleNet", SimpleNet, {}),
            ("ConfigurableNet", ConfigurableNet, {"hidden_size": 32}),
            ("ConvNet", ConvNet, {}),
            ("RNNModel", RNNModel, {"hidden_size": 64, "rnn_type": "LSTM"}),
        ]
        
        for name, model_class, kwargs in architectures:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                model = model_class(**kwargs)
                torch.save(model.state_dict(), tmp.name)
                
                with patch('load_models._model_class', model_class):
                    loaded_model = load_model_from_checkpoint(tmp.name)
                    assert isinstance(loaded_model, model_class)

                    norm = l2_norm_of_model(loaded_model)
                    assert isinstance(norm, float)
                
                os.unlink(tmp.name)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_checkpoint_directory(self):
        """Test handling of empty checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(Exception, match="could not find any valid model checkpoints"):
                contains_checkpoints(tmpdir)
    
    def test_model_with_no_trainable_params(self):
        """Test metrics on model with no trainable parameters."""
        model = NoParamsModel()
        norm = l2_norm_of_model(model)
        assert norm == 0.0
    



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/bin/bash
# Setup script to test phase-viz with improved loading indicators

# Set up directories
echo "Creating directories..."
mkdir -p large_checkpoints
mkdir -p src/temp

# Copy the updated files to the src directory
echo "Copying updated files to src directory..."
cp simplified_loading.py src/
cp smart_loader.py src/loader.py
cp updated_metrics.py src/metrics.py
cp updated_cli_simplified.py src/cli.py
cp large_model_simplified.py src/
cp custom_metrics.py src/

# Generate test checkpoints
echo "Generating test checkpoints..."
python generate_checkpoints_simplified.py

# Print instructions
echo -e "\n=== Phase-viz with Improved Loading Indicators ==="
echo "The system has been updated with:"
echo "1. Simplified loading animation showing single status for all metrics"
echo "2. Earlier display of usage instructions before opening the browser"
echo "3. Removed configuration file requirement - model parameters auto-detected"
echo "4. Sequential processing for clearer status updates"
echo -e "\nRun the tool with:"
echo "cd src"
echo "python cli.py load-dir --dir ../large_checkpoints --model large_model_simplified.py --class-name LargeNet"
echo -e "\nWhen prompted, enter 'custom_metrics.py' to test multiple metrics, or 'l2' for a single metric."

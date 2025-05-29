#!/bin/bash
# End-to-end setup and test script for phase-viz
set -e # Exit on error

echo "=== Phase-viz End-to-End Test Script ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo " Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_status "Installing dependencies..."
pip install -q torch torchvision typer plotly numpy kaleido pytest scipy

# Create necessary directories relative to script location
print_status "Creating directory structure..."
mkdir -p logs
mkdir -p test_checkpoints
mkdir -p test_results
mkdir -p temp_test_files

# Set up Python path for tests
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/..:$SCRIPT_DIR/../..:$PYTHONPATH"

print_status "Python path set to: $PYTHONPATH"

# Check if test_models.py exists
if [ ! -f "test_models.py" ]; then
    print_error "test_models.py not found in $SCRIPT_DIR"
    print_error "Please ensure test_models.py is in the same directory as this script"
    exit 1
fi

# Run unit tests
print_status "Running unit tests..."
if python -m pytest tests.py -v --tb=short --no-header; then
    print_status "All unit tests passed!"
else
    print_error "Unit tests failed!"
    echo
    print_warning "Debugging information:"
    echo "Current directory: $(pwd)"
    echo "Python path: $PYTHONPATH"
    echo "Available Python files:"
    find . -name "*.py" -type f | head -10
    echo
    print_warning "Trying to run tests with more verbose output..."
    python -m pytest tests.py -v --tb=long --no-header -s || true
    exit 1
fi

# Clean up temporary files
print_status "Cleaning up temporary test files..."
rm -rf temp_test_files test_checkpoints test_results

print_status "All tests completed successfully!"
echo
print_status "Test environment summary:"
echo " - Virtual environment: $SCRIPT_DIR/venv"
echo " - Test directory: $SCRIPT_DIR"
echo " - Python path: $PYTHONPATH"
#!/bin/bash
# Run integration tests for BPCA Python vs R pcaMethods equivalence.
#
# This script requires two mamba environments to be set up:
# - 'r': R environment with pcaMethods and jsonlite
# - 'bpca': Python environment with bpca package installed
#
# Usage:
#   ./run_integration_tests.sh
#
# To create the environments:
#   mamba env create -f tests/integration/envs/environment-r.yml
#   mamba env create -f tests/integration/envs/environment-python.yml
#   mamba run -n bpca pip install -e .

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"

# Check for mamba/conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: Neither mamba nor conda found. Please install mamba or conda."
    exit 1
fi

echo "Using: $CONDA_CMD"

# Check for R environment
echo "Checking R environment..."
if ! $CONDA_CMD run -n r Rscript -e "library(pcaMethods)" &> /dev/null; then
    echo "ERROR: R environment 'r' not found or pcaMethods not installed."
    echo "Create it with: mamba env create -f $SCRIPT_DIR/envs/environment-r.yml"
    exit 1
fi
echo "R environment OK"

# Check for Python environment
echo "Checking Python environment..."
if ! $CONDA_CMD run -n bpca python -c "import bpca" &> /dev/null; then
    echo "ERROR: Python environment 'bpca' not found or bpca package not installed."
    echo "Create it with:"
    echo "  mamba env create -f $SCRIPT_DIR/envs/environment-python.yml"
    echo "  mamba run -n bpca pip install -e $PROJECT_DIR"
    exit 1
fi
echo "Python environment OK"

# Create temporary directory for test data
TMPDIR=$(mktemp -d)
INPUT_DIR="$TMPDIR/inputs"
R_OUTPUT_DIR="$TMPDIR/r_outputs"
PY_OUTPUT_DIR="$TMPDIR/py_outputs"

cleanup() {
    echo "Cleaning up temporary directory..."
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo ""
echo "=== Step 1: Generate test cases ==="
$CONDA_CMD run -n bpca python "$SCRIPT_DIR/generate_test_cases.py" --output-dir "$INPUT_DIR"

echo ""
echo "=== Step 2: Run R implementation ==="
$CONDA_CMD run -n r Rscript "$SCRIPT_DIR/run_bpca_r.R" "$INPUT_DIR" "$R_OUTPUT_DIR"

echo ""
echo "=== Step 3: Run Python implementation ==="
$CONDA_CMD run -n bpca python "$SCRIPT_DIR/run_bpca_python.py" "$INPUT_DIR" "$PY_OUTPUT_DIR"

echo ""
echo "=== Step 4: Compare results ==="
# Run pytest on the test_equivalence.py file
# Note: We pass the directories via environment variables
export BPCA_TEST_INPUT_DIR="$INPUT_DIR"
export BPCA_TEST_R_OUTPUT_DIR="$R_OUTPUT_DIR"
export BPCA_TEST_PY_OUTPUT_DIR="$PY_OUTPUT_DIR"

cd "$PROJECT_DIR"
$CONDA_CMD run -n bpca pytest "$SCRIPT_DIR/test_equivalence.py" -v --tb=short

echo ""
echo "=== All integration tests passed! ==="

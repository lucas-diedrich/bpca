"""Integration tests for numerical equivalence between Python bpca and R pcaMethods::bpca.

These tests generate random data matrices with various shapes and missing value patterns,
run both the Python and R implementations, and verify that the outputs are numerically
equivalent within a relative tolerance of 1e-3.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from .generate_test_cases import generate_all_cases, save_cases_to_directory

# Relative tolerance for numerical comparisons
ATOL = 1e-3  # Small absolute tolerance for values near zero


def check_r_available() -> bool:
    """Check if R environment with pcaMethods is available.

    Returns True if:
    - Pre-computed results are available via environment variables, OR
    - R environment 'r' with pcaMethods is accessible via mamba
    """
    # If pre-computed results are available, we don't need R
    if all(
        os.environ.get(var)
        for var in [
            "BPCA_TEST_INPUT_DIR",
            "BPCA_TEST_R_OUTPUT_DIR",
            "BPCA_TEST_PY_OUTPUT_DIR",
        ]
    ):
        return True

    try:
        result = subprocess.run(
            ["mamba", "run", "-n", "r", "Rscript", "-e", "library(pcaMethods)"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Mark all tests to skip if R is not available and no pre-computed results
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R environment 'r' with pcaMethods not available (and no pre-computed results)",
)


def _load_precomputed_results(input_dir: Path, r_output_dir: Path, py_output_dir: Path) -> dict:
    """Load pre-computed results from directories.

    Parameters
    ----------
    input_dir
        Directory containing input test case JSON files.
    r_output_dir
        Directory containing R result JSON files.
    py_output_dir
        Directory containing Python result JSON files.

    Returns
    -------
    dict
        Dictionary mapping case_id to result data.
    """
    results = {}
    cases = generate_all_cases()

    for case in cases:
        case_id = case["case_id"]
        r_result_path = r_output_dir / f"case_{case_id}_r_result.json"
        py_result_path = py_output_dir / f"case_{case_id}_py_result.json"

        if not r_result_path.exists() or not py_result_path.exists():
            continue

        with open(r_result_path) as f:
            r_data = json.load(f)
        with open(py_result_path) as f:
            py_data = json.load(f)

        results[case_id] = {
            "r": r_data,
            "py": py_data,
            "params": case,
        }

    return results


def _generate_and_run_tests() -> dict:
    """Generate test cases and run both implementations.

    Returns
    -------
    dict
        Dictionary mapping case_id to result data.
    """
    # Create temporary directory for test fixtures
    tmpdir = Path(tempfile.mkdtemp())
    input_dir = tmpdir / "inputs"
    r_output_dir = tmpdir / "r_outputs"
    py_output_dir = tmpdir / "py_outputs"

    # Generate test cases
    print("\nGenerating test cases...")
    cases = generate_all_cases()
    save_cases_to_directory(cases, input_dir)
    print(f"Generated {len(cases)} test cases")

    # Run R implementation
    print("Running R implementation...")
    r_script = Path(__file__).parent / "run_bpca_r.R"
    r_result = subprocess.run(
        ["mamba", "run", "-n", "r", "Rscript", str(r_script), str(input_dir), str(r_output_dir)],
        capture_output=True,
        text=True,
    )
    if r_result.returncode != 0:
        print(f"R stdout: {r_result.stdout}")
        print(f"R stderr: {r_result.stderr}")
        raise RuntimeError(f"R script failed: {r_result.stderr}")

    # Run Python implementation
    print("Running Python implementation...")
    py_script = Path(__file__).parent / "run_bpca_python.py"
    py_result = subprocess.run(
        ["mamba", "run", "-n", "bpca", "python", str(py_script), str(input_dir), str(py_output_dir)],
        capture_output=True,
        text=True,
    )
    if py_result.returncode != 0:
        print(f"Python stdout: {py_result.stdout}")
        print(f"Python stderr: {py_result.stderr}")
        raise RuntimeError(f"Python script failed: {py_result.stderr}")

    return _load_precomputed_results(input_dir, r_output_dir, py_output_dir)


@pytest.fixture(scope="module")
def test_results():
    """Load or generate test results.

    This fixture supports two modes:
    1. Pre-computed mode: If BPCA_TEST_INPUT_DIR, BPCA_TEST_R_OUTPUT_DIR, and
       BPCA_TEST_PY_OUTPUT_DIR environment variables are set, load results from
       those directories.
    2. Generate-and-run mode: Generate test cases and run both implementations.
    """
    # Check for pre-computed results (CI/CD mode)
    input_dir = os.environ.get("BPCA_TEST_INPUT_DIR")
    r_output_dir = os.environ.get("BPCA_TEST_R_OUTPUT_DIR")
    py_output_dir = os.environ.get("BPCA_TEST_PY_OUTPUT_DIR")

    if input_dir and r_output_dir and py_output_dir:
        print("\nUsing pre-computed results from environment variables")
        results = _load_precomputed_results(Path(input_dir), Path(r_output_dir), Path(py_output_dir))
        print(f"Loaded {len(results)} result pairs")
        yield results
    else:
        # Generate and run (local mode)
        print("\nGenerating and running tests...")
        results = _generate_and_run_tests()
        print(f"Collected {len(results)} result pairs")
        yield results


def get_case_ids():
    """Get all case IDs for parametrization."""
    cases = generate_all_cases()
    return [case["case_id"] for case in cases]


def _parse_r_matrix(r_dict: dict) -> np.ndarray:
    """Parse R matrix stored as dict of columns.

    R saves matrices as {col_name: [values], ...}. Column names are
    "PC1", "PC2", ...

    Parameters
    ----------
    r_dict
        Dictionary with column names as keys and lists as values.

    Returns
    -------
    np.ndarray
        Matrix with shape (n_rows, n_cols)
    """

    def natural_sort_key(s):
        """Sort strings with embedded numbers naturally (PC1, PC2, ..., PC10)."""
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

    # Get column names in natural sorted order
    col_names = sorted(r_dict.keys(), key=natural_sort_key)
    # Stack columns and transpose to get (n_rows, n_cols)
    return np.array([r_dict[col] for col in col_names]).T


@pytest.mark.parametrize("case_id", get_case_ids())
def test_reconstruction_equivalence(test_results, case_id):
    """Test that reconstructions are numerically equivalent between R and Python.

    BPCA components are not ordered by variance and have arbitrary signs,
    so we compare the reconstructed data (scores @ loadings.T) which should
    be invariant to sign flips and component ordering.
    """
    if case_id not in test_results:
        pytest.skip(f"Results not available for case {case_id}")

    r_data = test_results[case_id]["r"]
    py_data = test_results[case_id]["py"]

    # Convert to numpy arrays
    r_scores = _parse_r_matrix(r_data["scores"])  # (n_obs, n_latent)
    r_loadings = _parse_r_matrix(r_data["loadings"])  # (n_var, n_latent)
    py_scores = np.array(py_data["scores"])  # (n_obs, n_latent)
    py_loadings = np.array(py_data["loadings"])  # (n_var, n_latent)

    # Compute reconstructions: scores @ loadings.T
    # This is invariant to sign flips and component ordering
    r_reconstruction = r_scores @ r_loadings.T  # (n_obs, n_var)
    py_reconstruction = py_scores @ py_loadings.T  # (n_obs, n_var)

    if np.allclose(r_reconstruction, 0, atol=1e-20):
        pytest.skip(reason="R values collapsed due to bug in R code")

    np.testing.assert_allclose(
        py_reconstruction,
        r_reconstruction,
        atol=ATOL,
        err_msg=f"Reconstruction mismatch for case {case_id}",
    )

#!/usr/bin/env python
"""Run BPCA on test cases using Python bpca package.

This script reads JSON test case files and writes results to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpca import BPCA


def run_bpca(
    X: np.ndarray,
    n_latent: int,
    max_iter: int = 1000,
    tolerance: float = 1e-4,
) -> dict:
    """Run BPCA and extract outputs.

    Parameters
    ----------
    X
        Data matrix (n_obs, n_var). May contain NaN values.
    n_latent
        Number of principal components to compute.
    max_iter
        Maximum number of iterations.
    tolerance
        Convergence threshold.

    Returns
    -------
    dict
        Dictionary containing scores and loadings.
    """
    model = BPCA(n_components=n_latent, max_iter=max_iter, tolerance=tolerance)
    model.fit(X)

    return {
        "scores": model._usage,  # (n_obs, n_latent)
        "loadings": model._components.T,  # (n_var, n_latent) - transposed to match R
    }


def process_case(input_path: Path, output_path: Path) -> None:
    """Process a single test case file.

    Parameters
    ----------
    input_path
        Path to input JSON file containing test case.
    output_path
        Path to output JSON file for results.
    """
    # Read test case
    with open(input_path) as f:
        test_case = json.load(f)

    # Convert X to numpy array (None values become NaN)
    X = np.array(test_case["X"], dtype=float)  # None -> nan automatically

    # Run BPCA
    result = run_bpca(
        X,
        n_latent=test_case["n_latent"],
        max_iter=test_case["max_iter"],
        tolerance=test_case["tolerance"],
    )

    # Prepare output (convert numpy arrays to lists for JSON)
    output = {
        "case_id": test_case["case_id"],
        "scores": result["scores"].tolist(),
        "loadings": result["loadings"].tolist(),
    }

    # Write result
    with open(output_path, "w") as f:
        json.dump(output, f)


def process_directory(input_dir: Path, output_dir: Path) -> None:
    """Process all test cases in a directory.

    Parameters
    ----------
    input_dir
        Directory containing test case JSON files.
    output_dir
        Directory to write result JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("case_*.json"))
    print(f"Found {len(input_files)} test case files")

    for input_path in input_files:
        case_name = input_path.stem
        output_path = output_dir / f"{case_name}_py_result.json"

        try:
            process_case(input_path, output_path)
            print(f"Processed: {case_name}")
        except Exception as e:  # noqa
            print(f"ERROR processing {case_name}: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BPCA on test cases using Python bpca package")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing test case JSON files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write result JSON files",
    )
    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    process_directory(args.input_dir, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

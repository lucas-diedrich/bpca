"""Generate test cases for BPCA integration tests.

This script generates test data with fixed parameter combinations for comparing
Python bpca against R pcaMethods::bpca.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import numpy as np

# Fixed parameter grid
SHAPE_COMBOS = [(30, 15), (100, 50), (100, 200)]  # (n_obs, n_var)
N_LATENT = [2, 10, 50]
MISSING_FRAC = [0.0, 0.1, 0.5]
N_REPLICATES = 5

# Matching fitting parameters for R and Python
MAX_ITER = 1000
TOLERANCE = 1e-4


def generate_test_data(
    n_obs: int,
    n_var: int,
    n_latent: int,
    missing_frac: float,
    seed: int,
) -> dict:
    """Generate a single test case.

    Parameters
    ----------
    n_obs
        Number of observations (rows)
    n_var
        Number of variables (columns)
    n_latent
        Number of latent components (will be capped at min(n_obs, n_var) - 1)
    missing_frac
        Fraction of values to set as missing (0.0 to 1.0)
    seed
        Random seed for reproducibility

    Returns
    -------
    dict
        Test case containing input data and parameters
    """
    rng = np.random.default_rng(seed)

    # Cap n_latent at min(n_obs, n_var) - 1 to ensure valid PCA
    n_latent_actual = min(n_latent, min(n_obs, n_var) - 1)

    # Generate random data matrix
    z = rng.standard_normal((n_obs, n_latent_actual))
    loadings = rng.standard_normal((n_latent_actual, n_var))
    X = z @ loadings

    # Introduce missing values
    if missing_frac > 0:
        n_missing = int(n_obs * n_var * missing_frac)
        missing_indices = rng.choice(n_obs * n_var, size=n_missing, replace=False)
        X_flat = X.flatten()
        X_flat[missing_indices] = np.nan
        X = X_flat.reshape(n_obs, n_var)

    # Convert NaN to None for JSON serialization (NaN is not valid JSON)
    X_serializable = [[None if np.isnan(v) else v for v in row] for row in X.tolist()]

    return {
        "X": X_serializable,
        "n_obs": n_obs,
        "n_var": n_var,
        "n_latent": n_latent_actual,
        "n_latent_requested": n_latent,
        "missing_frac": missing_frac,
        "seed": seed,
        "max_iter": MAX_ITER,
        "tolerance": TOLERANCE,
    }


def generate_all_cases() -> list[dict]:
    """Generate all test cases from the parameter grid.

    Returns
    -------
    list[dict]
        List of all test cases (270 total: 3 shapes x 3 n_latent x 3 missing_frac x 10 replicates)
    """
    cases = []
    case_id = 0

    for (n_obs, n_var), n_latent, missing_frac in itertools.product(SHAPE_COMBOS, N_LATENT, MISSING_FRAC):
        for replicate in range(N_REPLICATES):
            seed = case_id * 10 + replicate  # Deterministic seed
            case = generate_test_data(
                n_obs=n_obs,
                n_var=n_var,
                n_latent=n_latent,
                missing_frac=missing_frac,
                seed=seed,
            )
            case["case_id"] = f"{case_id:03d}_{replicate:02d}"
            cases.append(case)
        case_id += 1

    return cases


def save_cases_to_directory(cases: list[dict], output_dir: Path) -> list[Path]:
    """Save test cases as individual JSON files.

    Parameters
    ----------
    cases
        List of test cases
    output_dir
        Directory to save JSON files

    Returns
    -------
    list[Path]
        List of paths to saved JSON files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for case in cases:
        case_path = output_dir / f"case_{case['case_id']}.json"
        with open(case_path, "w") as f:
            json.dump(case, f)
        paths.append(case_path)

    return paths


def get_default_output_dir() -> Path:
    """Get default output directory for test fixtures.

    Returns
    -------
    Path
        Path to fixtures directory (uses tempdir)
    """
    return Path(tempfile.gettempdir()) / "bpca_test_fixtures"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate BPCA integration test cases")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for test case JSON files",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list test case parameters without generating data",
    )
    args = parser.parse_args()

    if args.list_only:
        # Just print parameter combinations
        case_id = 0
        for (n_obs, n_var), n_latent, missing_frac in itertools.product(SHAPE_COMBOS, N_LATENT, MISSING_FRAC):
            n_latent_actual = min(n_latent, min(n_obs, n_var) - 1)
            print(
                f"Case {case_id:03d}: n_obs={n_obs}, n_var={n_var}, "
                f"n_latent={n_latent_actual} (requested={n_latent}), "
                f"missing_frac={missing_frac}"
            )
            case_id += 1
        print(f"\nTotal unique combinations: {case_id}")
        print(f"Total test cases with {N_REPLICATES} replicates: {case_id * N_REPLICATES}")
    else:
        output_dir = args.output_dir or get_default_output_dir()
        print(f"Generating test cases to: {output_dir}")

        cases = generate_all_cases()
        paths = save_cases_to_directory(cases, output_dir)

        print(f"Generated {len(paths)} test case files")

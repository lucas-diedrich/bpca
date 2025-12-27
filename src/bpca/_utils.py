"""Utility module"""

from typing import Literal

import numpy as np


def impute_missing(x: np.ndarray, strategy: Literal["median", "zero"] = "zero") -> np.ndarray:
    """Impute missing values

    Parameters
    ----------
    x
        Matrix (n_obs, n_vars) with missing values
    strategy
        Imputation strategy
            - `zero`: Impute zeros. Strategy in `pcaMethods`
            - `median`: Impute feature-wise median of non-missing observations
    """
    missing_mask = np.isnan(x)
    if not missing_mask.any():
        return x

    if strategy == "median":
        feature_medians = np.nanmedian(x, axis=0, keepdims=True)
        return np.where(missing_mask, feature_medians, x)
    elif strategy == "zero":
        return np.where(missing_mask, 0, x)
    else:
        raise ValueError(f"`strategy` must be one of ('zero', 'median'), not {strategy}")


def compute_variance_explained(X: np.ndarray, usage: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Compute variance explained by each component.

    Uses leave-one-out contributions normalized to sum to total R².
    This handles non-orthogonal components correctly.

    Parameters
    ----------
    X
        Original data matrix (n_obs, n_var)
    usage
        Score matrix (n_obs, n_components)
    loadings
        Loading matrix (n_components, n_var)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Variance explained per component (sorted descending) and sort order
    """
    X_centered = X - np.nanmean(X, axis=0)
    total_ss = np.nansum(np.square(X_centered))

    # Full reconstruction with all components
    X_full = usage @ loadings
    full_residual_ss = np.nansum(np.square(X_centered - X_full))
    total_r2 = 1 - full_residual_ss / total_ss

    # Compute relative contribution of each component (leave-one-out)
    n_components = usage.shape[1]
    contributions = np.zeros(n_components)

    for k in range(n_components):
        # Reconstruction without component k
        mask = np.ones(n_components, dtype=bool)
        mask[k] = False
        X_without_k = usage[:, mask] @ loadings[mask, :]
        residual_without_k = np.nansum(np.square(X_centered - X_without_k))

        # Raw contribution of component k
        contributions[k] = residual_without_k - full_residual_ss

    return contributions / np.nansum(contributions) * total_r2

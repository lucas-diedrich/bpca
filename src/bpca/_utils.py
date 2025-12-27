"""Utility module"""

import numpy as np


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

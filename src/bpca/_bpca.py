"""High-level API"""

from typing import Literal

import numpy as np

from ._core import BPCAFit


class BPCA:
    """Bayesian principal component analysis"""

    def __init__(
        self,
        n_components: int | None = None,
        sigma2_init: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        init_w: Literal["svd", "random"] = "random",
    ):
        """Initialize Bayesian principal component analysis"""
        self.n_components = n_components
        self.sigma2_init = sigma2_init
        self.max_iter = max_iter
        self.tol = tol
        self.init_w = init_w

        self._components = None
        self._usage = None
        self._alpha = None
        self._sigma2 = None
        self._mu = None

        self._explained_variance_ratio_ = None

        self._n_iter = None

        self._is_fit = False

    def compute_variance_explained(
        self, X: np.ndarray, usage: np.ndarray, loadings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

        # Normalize contributions to sum to total R²
        if contributions.sum() > 0:
            var_explained = contributions / contributions.sum() * total_r2
        else:
            var_explained = np.zeros(n_components)

        order = np.argsort(var_explained)[::-1]
        return var_explained[order], order

    def fit(self, X: np.ndarray) -> "BPCA":
        """Fit parameters.

        Parameters
        ----------
        X
            Data matrix (n_obs, n_var)

        Returns
        -------
        BPCA
            Fitted model (self)
        """
        bpca = BPCAFit(
            X.T,
            n_latent=self.n_components,
            sigma2=self.sigma2_init,
            max_iter=self.max_iter,
            tol=self.tol,
            init_w=self.init_w,
        )
        bpca.fit()

        self._mu = bpca.mu
        self._alpha = bpca.alpha  # (n_latent,)
        usage = bpca.z.T  # (n_latent, n_obs) -> (n_obs, n_latent)
        loadings = bpca.w.T  # (n_var, n_latent) -> (n_latent, n_var)
        self._explained_variance_ratio_, dim_order = self.compute_variance_explained(
            X=X, usage=usage, loadings=loadings
        )

        self._components = self.sort_dimensions(loadings, order=dim_order, axis=0)
        self._usage = self.sort_dimensions(usage, order=dim_order, axis=1)
        self._sigma2 = bpca.sigma2
        self._n_iter = bpca.n_iter

        self._is_fit = True

        return self

    def sort_dimensions(self, x: np.ndarray, order: np.ndarray, axis: int) -> np.ndarray:
        """Sort array dimensions based on on external ranking value (increasing order) along axis"""
        return np.take(x, indices=order, axis=axis)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted model.

        Parameters
        ----------
        X
            Data matrix (n_obs, n_var)

        Returns
        -------
        np.ndarray
            Transformed data (n_obs, n_components)
        """
        self._check_is_fit()
        X_centered = X - self._mu
        return X_centered @ self._components.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return transformed training data.

        Parameters
        ----------
        X
            Data matrix (n_obs, n_var)

        Returns
        -------
        np.ndarray
            Transformed data (n_obs, n_components)
        """
        self.fit(X)
        return self._usage

    def _check_is_fit(self) -> None:
        """Check whether model is fit

        Raises
        ------
        RuntimeError
            If model is not yet fit.
        """
        if not self._is_fit:
            raise RuntimeError("Fit model first.")

    @property
    def components_(self) -> np.ndarray:
        """Principal axes in feature space (n_components_, n_features)"""
        self._check_is_fit()
        return self._components

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of variance explained by each of the selected components (n_components_,)"""
        self._check_is_fit()
        return self._explained_variance_ratio_

    @property
    def n_components_(self) -> int:
        """Number of components (n_components_,)"""
        return self.n_components

    @property
    def n_iter(self) -> int:
        """Number of iterations"""
        self._check_is_fit()
        return self._n_iter

    @property
    def alpha(self) -> np.ndarray:
        r"""Estimated regularization strength :math:`\alpha^2` by EM algorithm."""
        self._check_is_fit()
        return np.sort(self._alpha)

    @property
    def sigma2(self) -> float:
        r"""Estimated variance :math:`\sigma^2` by EM algorithm."""
        self._check_is_fit()
        return self._sigma2

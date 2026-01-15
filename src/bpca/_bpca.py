"""High-level API"""

import numpy as np

from ._core import BPCAFit
from ._utils import compute_variance_explained


class BPCA:
    """Bayesian principal component analysis (BPCA)

    Implements the BPCA method (generative model suggested by Bishop, 1998) as suggested by Oba et al (2003).
    The implementation follows the reference implementation in R (Stacklies, 2007).

    Examples
    --------

    .. code:: python

        from bpca import BPCA
        from sklearn.datasets import load_iris

        iris_dataset = load_iris()
        X = iris_dataset["data"] # (n_obs, n_var)
        bpca = BPCA(n_components=None)

        usage = bpca.fit_transform(X) # (n_components, n_latent)
        weights = bpca.components_  # (n_latent, n_var)

    Citation
    --------
    - Bishop, C. Bayesian PCA. in Advances in Neural Information Processing Systems vol. 11 (MIT Press, 1998).
    - Oba, S. et al. A Bayesian missing value estimation method for gene expression profile data. Bioinformatics 19, 2088 - 2096 (2003).
    - Stacklies, W., Redestig, H., Scholz, M., Walther, D. & Selbig, J. pcaMethods—a bioconductor package providing PCA methods for incomplete data. Bioinformatics 23, 1164 - 1167 (2007).
    """

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
        *,
        sort_components: bool = True,
    ):
        """Initialize Bayesian principal component analysis

        Parameters
        ----------
        n_components
            Number of components to compute. If `None`, uses n_var - 1 dimensions
        max_iter
            Maximum number of EM iterations
        tolerance
            Convergence tolerance
        sort_components
            Whether to sort the components and parameters by decreasing explained variance (`True`) or leave them unsorted (`False`). Defaults to `True`.
        """
        self._n_components = n_components
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._sort_components = sort_components

        self._is_fit = False

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
            X,
            n_latent=self._n_components,
            max_iter=self._max_iter,
            tolerance=self._tolerance,
        )
        bpca.fit()

        self._mu = bpca.mu
        self._alpha = bpca.alpha.flatten()  # (n_latent,)
        self._usage = bpca.z  # (n_latent, n_obs) -> (n_obs, n_latent)
        self._components = bpca.weights.T  # (n_var, n_components) -> (n_components, n_var)
        self._tau = float(bpca.tau.squeeze())
        self._n_iter = bpca.n_iter

        self._explained_variance_ratio_ = compute_variance_explained(
            X=X,
            usage=self._usage,
            loadings=self._components,
        )

        if self._sort_components:
            self._sort_by_explained_variance()

        self._is_fit = True

        return self

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

    def _sort_by_explained_variance(self) -> None:
        """Sort all component arrays by explained variance in descending order."""
        sort_idx = np.argsort(self._explained_variance_ratio_)[::-1]
        self._explained_variance_ratio_ = self._explained_variance_ratio_[sort_idx]
        self._alpha = self._alpha[sort_idx]
        self._components = self._components[sort_idx, :]
        self._usage = self._usage[:, sort_idx]

    @property
    def components_(self) -> np.ndarray:
        """Principal axes in feature space (``n_components_``, ``n_features``)."""
        self._check_is_fit()
        return self._components

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of variance explained by each of the selected components (``n_components_``,)."""
        self._check_is_fit()
        return self._explained_variance_ratio_

    @property
    def n_components_(self) -> int:
        """Number of components (``n_components_``,)."""
        return self._n_components

    @property
    def n_iter(self) -> int:
        """Number of iterations"""
        self._check_is_fit()
        return self._n_iter

    @property
    def alpha(self) -> np.ndarray:
        r"""Estimated regularization strength :math:`\alpha^2` by EM algorithm."""
        self._check_is_fit()
        return self._alpha

    @property
    def tau(self) -> float:
        r"""Estimated variance :math:`\sigma^2` by EM algorithm."""
        self._check_is_fit()
        return self._tau

    @property
    def mu(self) -> np.ndarray:
        r"""Estimated feature-wise mean :math:`\mu` by EM algorithm."""
        self._check_is_fit()
        return self._mu.reshape(-1)

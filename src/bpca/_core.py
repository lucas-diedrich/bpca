"""Core algorithm"""

import warnings
from typing import Literal

import numpy as np


class ConvergenceWarning(Warning):
    """Algorithm did not converge"""


def _impute_missing(x: np.ndarray, strategy: Literal["median", "zero"] = "zero") -> np.ndarray:
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
        feature_medians = np.nanmedian(x, axis=1, keepdims=True)
        return np.where(missing_mask, feature_medians, x)
    elif strategy == "zero":
        return np.where(missing_mask, 0, x)
    else:
        raise ValueError(f"`strategy` must be one of ('zero', 'median'), not {strategy}")


class BPCAFit:
    r"""Bayesian principal component analysis fitting procedure

    Fits the model with an EM-procedure

    Initialization
        - W: Random (randu)
        - sigma2: 1
        - alpha (n_latent,): Random (randu)
        - var: Sigma2 x Identity matrix
        - Z: Zeros

    E step
        - <xn> = M-1 x W.T x (X - t)

    M step
        - Update W
        - Update sigma
        - Update alpha

    Examples
    --------

    .. code:: python

        from bpca._core import BPCAFit
        from sklearn.datasets import load_iris

        iris_dataset = load_iris()
        X = iris_dataset["data"] # (n_obs, n_var)

        # Expects (n_var, n_obs)
        bpca = BPCAFit(X=X.T)
        bpca.fit()

    Citation
    --------
    Bishop, C. Bayesian PCA. in Advances in Neural Information Processing Systems vol. 11 (MIT Press, 1998).
    """

    _INIT_W_OPTIONS = ("svd", "random")
    _IMPUTATION_OPTIONS = ("zero", "median")

    def __init__(
        self,
        X: np.ndarray,
        n_latent: int | None = 50,
        sigma2: float = 1.0,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        weight_init_strategy: Literal["svd", "random"] = "random",
    ) -> None:
        """Initialize Fit

        Parameters
        ----------
        X
            (n_var, n_obs)
        n_latent
            Number of latent dimensions to consider. If `None`, uses n_var - 1 dimensions
        alpha
            ARD prior strength
        sigma2
            Variance prior
        max_iter
            Maximum number of EM iterations
        tolerance
            Convergence tolerance
        weight_init_strategy
            How to initialize weight matrix (svd: PCA on zero-imputed data, random: random uniform 0-1 weight initialization)
        """
        if weight_init_strategy not in self._INIT_W_OPTIONS:
            raise ValueError(f"init_w must be one of {self._INIT_W_OPTIONS}, not {weight_init_strategy}")

        self.X = X

        # Parameters
        self._initialize_em_parameters(
            self.X, n_latent=n_latent, sigma2=sigma2, weight_init_strategy=weight_init_strategy
        )
        self._initialize_fit_procedure_parameters(max_iter=max_iter, tolerance=tolerance)
        self._initialize_fit_result_parameters()

    def _initialize_em_parameters(
        self, X: np.ndarray, n_latent: int, sigma2: float, weight_init_strategy: Literal["random", "svd"] = "svd"
    ):
        """Initialize parameters for EM algorithm

        Initializes
            - shape parameters `n_var`, `n_obs`
            - Number of latent dimensions `n_latent`
            - mean `mu`
            - mean-centered data `Xt`
            - Latent variable `z`
            - loadings `weights`
            - unexplained variance matrix `var`
            - ARD prior `alpha`

        Parameters
        ----------
        X
            Data (n_observations, n_features)
        n_latent
            Number of latent dimensions
        sigma2
            Variance
        weight_init_strategy
            How to initialize weights
        """
        self.n_var, self.n_obs = X.shape

        if n_latent is not None and n_latent > X.shape[0] - 1:
            warnings.warn(
                f"n_latent={n_latent} is larger than number of array dimensions ({X.shape[0]}). Set to maximum number {X.shape[0] - 1}",
                stacklevel=2,
            )
        self.n_latent = min(n_latent, X.shape[0], X.shape[1]) if n_latent is None else X.shape[0]

        self.mu = np.nanmean(self.X, axis=1, keepdims=True)  # (n_features, 1)
        self.Xt = X - self.mu

        self.var = sigma2 * np.eye(self.n_latent)  # (n_latent, n_latent)

        self.z = None
        self.weights = (
            np.random.rand(self.n_var, self.n_latent)
            if self.init_w == "random"
            else self._svd_initialize_weights(x=self.Xt, n_latent=self.n_latent)
        )  # (n_features, n_latent)

        # TODO: Evaluate unexplained variance estimation via PCA

        self.alpha = np.random.rand(self.n_latent)  # (n_latent,)

    def _svd_initialize_weights(self, X: np.ndarray, n_latent: int, strategy: str = "zero") -> np.ndarray:
        """SVD initialize data

        Parameters
        ----------
        x
            (n_obs, n_var)
        n_latent
            Number of latent dimensions to consider
        """
        X = _impute_missing(X, strategy=strategy)
        covariance_matrix = X @ X.T
        U, S, _ = np.linalg.svd(covariance_matrix, full_matrices=False)

        return U.T[:, :n_latent] * S.T[:n_latent]  # or just U[:, :q]

    def _initialize_fit_procedure_parameters(self, max_iter: int, tolerance: float) -> None:
        """Initialize paramters of the fitting procedure

        Initalizes `max_iter`, `tolerance`
        """
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _initialize_fit_result_parameters(self) -> None:
        """Initialize parameters that are used to evaluate the fitting procedure

        Initializes `_converged`, `_n_iter`, `_is_fit`
        """
        self._converged = None
        self._n_iter = None
        self._is_fit = False

    def fit(self):
        """Fit model"""
        converged = False

        tolgap = ...
        zn = ...
        weights = ...

        for n_iter in range(self.max_iter):  # noqa: B007
            self._e_step()
            self._m_step()

            if tolgap < self.tolerance:
                converged = True
                break

        if not converged:
            warnings.warn(f"Algorithm did not converge after {self.max_iter} steps", ConvergenceWarning, stacklevel=2)

        # Store final latent representation
        self.z = zn
        self.weights = weights

        self._converged = converged
        self._n_iter = n_iter + 1
        self._is_fit = True

    def _e_step(self, M: np.ndarray) -> np.ndarray:
        r"""Expectation step

        Computes the posterior mean of the latent variables z.

        Parameters
        ----------
        M : np.ndarray
            Precision matrix :math:`W^T W + \sigma^2 I` of shape (n_latent, n_latent)

        Returns
        -------
        zn : np.ndarray
            Posterior mean :math:`E[z|x]` of shape (n_latent, n_obs)
        """

    def _m_step(self, zn: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Maximization step

        Finds parameters that maximize the expected loglikelihood.

        Parameters
        ----------
        zn : np.ndarray
            Posterior mean E[z|x] of shape (n_latent, n_obs)
        M : np.ndarray
            Precision matrix W.T @ W + σ²I of shape (n_latent, n_latent)

        Returns
        -------
        M_new : np.ndarray
            Updated precision matrix for next iteration
        """

    def _convergence_criterium(self):
        """Convergence criterium"""

    def _check_is_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("Run .fit() method first")

    @property
    def n_iter(self) -> int:
        """Number of iterations until convergence"""
        return self._n_iter

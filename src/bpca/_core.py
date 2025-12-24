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
        feature_medians = np.nanmedian(x, axis=0, keepdims=True)
        return np.where(missing_mask, feature_medians, x)
    elif strategy == "zero":
        return np.where(missing_mask, 0, x)
    else:
        raise ValueError(f"`strategy` must be one of ('zero', 'median'), not {strategy}")


class BPCAFit:
    r"""Bayesian principal component analysis fitting procedure

    Fits the model with an EM-procedure

    1. Initialization
    2. Run until convergence
        1. E-step (latent variable z computation)
        2. M-Step (update weights, ARD parameter alpha, unexplained variance sigma)
    3. Report

    Examples
    --------

    .. code:: python

        from bpca._core import BPCAFit
        from sklearn.datasets import load_iris

        iris_dataset = load_iris()
        X = iris_dataset["data"] # (n_obs, n_var)
        bpca = BPCAFit(X=X)
        bpca.fit()

    Citation
    --------
    Bishop, C. Bayesian PCA. in Advances in Neural Information Processing Systems vol. 11 (MIT Press, 1998).
    """

    _IMPUTATION_OPTIONS = ("zero", "median")

    MIN_RESIDUAL_VARIANCE = 1e-10
    MAX_RESIDUAL_VARIANCE = 1e10

    GAMMA_ALPHA0 = 1e-10
    """Uninformed prior for alpha parameter of gamma distribution"""
    GAMMA_BETA0 = 1.0
    """Uninformed prior for beta parameter of gamma distribution"""

    def __init__(
        self,
        X: np.ndarray,
        n_latent: int | None = 50,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """Initialize Fit

        Parameters
        ----------
        X
            (n_obs, n_var)
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
        """
        self.X = X

        # Parameters
        self._initialize_em_parameters(self.X, n_latent=n_latent)
        self._initialize_fit_procedure_parameters(max_iter=max_iter, tolerance=tolerance)
        self._initialize_fit_result_parameters()

    def _initialize_em_parameters(
        self,
        X: np.ndarray,
        n_latent: int,
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
        self.n_obs, self.n_var = X.shape

        if n_latent is not None and n_latent > min(X.shape):
            warnings.warn(
                f"n_latent={n_latent} is larger than number of array dimensions ({X.shape}). Set to maximum number {min(X.shape)}",
                stacklevel=2,
            )
        self.n_latent = min(n_latent, X.shape[0], X.shape[1]) if n_latent is not None else X.shape[1] - 1

        self.mu = np.nanmean(self.X, axis=0, keepdims=True)  # (n_features, 1)
        self.Xt = X - self.mu

        self.z = None
        self.weights, residual_variance = self._pca(X=self.Xt, n_latent=self.n_latent)  # (n_features, n_latent), float
        self.tau = 1 / np.clip(residual_variance, self.MIN_RESIDUAL_VARIANCE, self.MAX_RESIDUAL_VARIANCE)

        self.var = np.eye(self.n_latent)  # (n_latent, n_latent)

        self.alpha = (
            np.divide(
                2 * self.GAMMA_ALPHA0 + self.n_var,
                self.tau * np.sum(np.square(self.weights), axis=0) + 2 * self.GAMMA_ALPHA0 / self.GAMMA_BETA0,
            )  # (n_latent, )
        )

    def _pca(self, X: np.ndarray, n_latent: int, strategy: str = "zero") -> tuple[np.ndarray, float]:
        """Run PCA on imputed data

        Parameters
        ----------
        x
            (n_obs, n_var), mean-centered data
        n_latent
            Number of latent dimensions to consider

        Returns
        -------
        weights, residual_variance
            - Weights (n_features, n_latent)
            - Residual unexplained variance by SVD
        """
        X = _impute_missing(X, strategy=strategy)
        covariance_matrix = (X.T @ X) / (X.shape[0] - 1)
        U, S, _ = np.linalg.svd(covariance_matrix, full_matrices=False)

        residual_variance = np.trace(covariance_matrix) - np.sum(S[:n_latent])

        return U[:, :n_latent] * np.sqrt(S).T[:n_latent], residual_variance

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

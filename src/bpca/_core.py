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
    """Uninformed prior for gamma parameter of gamma distribution for alpha parameter"""
    BETA_ALPHA0 = 1.0
    """Uninformed prior for beta parameter of gamma distribution for alpha parameter"""

    GAMMA_TAU0 = 1e-10
    """Uninformed prior for gamma parameter of gamma distribution for tau parameter"""
    BETA_TAU0 = 1.0
    """Uninformed prior for beta parameter of gamma distribution for tau parameter"""

    GAMMA_MU0 = 0.001
    """Hyperparameter for tau update"""

    def __init__(
        self,
        X: np.ndarray,
        n_latent: int | None = 50,
        max_iter: int = 1000,
        tolerance: float = 1e-4,
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
            - Latent representation `z`
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

        self.nan_mask = np.isnan(X)  # (n_obs, n_var)
        self.complete_obs_idx = np.where(~self.nan_mask.any(axis=1))[0]
        self.incomplete_obs_idx = np.where(self.nan_mask.any(axis=1))[0]

        self.mu = np.nanmean(self.X, axis=0, keepdims=True)  # (1, n_var)
        self.Xt = X - self.mu

        # Initialize imputed data estimate (updated during E-step)
        self.X_imputed = np.where(self.nan_mask, 0, X)

        self.z = None
        self.weights, residual_variance = self._pca(X=self.Xt, n_latent=self.n_latent)  # (n_features, n_latent), float
        self.tau: float = 1 / np.clip(residual_variance, self.MIN_RESIDUAL_VARIANCE, self.MAX_RESIDUAL_VARIANCE)

        self.var = np.eye(self.n_latent)  # (n_latent, n_latent)

        self.alpha = (
            np.divide(
                2 * self.GAMMA_ALPHA0 + self.n_var,
                self.tau * np.sum(np.square(self.weights), axis=0) + 2 * self.GAMMA_ALPHA0 / self.BETA_ALPHA0,
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
            - Weights (n_latent, n_features)
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
        previous_tau = -np.inf
        for n_iter in range(self.max_iter):  # noqa: B007
            scores, T, trS, Rx = self._e_step()
            self._m_step(T, trS, Rx)

            delta_tau = abs(np.log10(self.tau) - np.log10(previous_tau))
            if delta_tau < self.tolerance:
                converged = True
                break
            previous_tau = self.tau

        self.z = scores

        if not converged:
            warnings.warn(f"Algorithm did not converge after {self.max_iter} steps", ConvergenceWarning, stacklevel=2)

        self._converged = converged
        self._n_iter = n_iter + 1
        self._is_fit = True

    def _e_step(self) -> tuple[np.ndarray, np.ndarray, float]:
        r"""Expectation step

        Computes the posterior mean of the latent variables z and sufficient
        statistics for the M-step.

        Returns
        -------
        scores
            Posterior mean E[z|x] of shape (n_obs, n_latent)
        T
            Cross-covariance sufficient statistic of shape (n_var, n_latent)
        trS
            Sum of squared residuals sufficient statistic
        """
        # Initialize scores matrix
        scores = np.full((self.n_obs, self.n_latent), np.nan)

        # Posterior precision matrix p(z|x) - shared for all complete observations
        # R line 19: Rx <- diag(M$comps) + M$tau * t(M$PA) %*% M$PA + M$SigW
        Rx = np.eye(self.n_latent) + self.tau * self.weights.T @ self.weights + self.var
        Rx_inv = np.linalg.inv(Rx)

        # --- E-step for complete observations (R lines 23-37) ---
        if len(self.complete_obs_idx) > 0:
            # Centered data for complete observations: (n_complete, n_var)
            # R line 27: dy <- y[idx,, drop=FALSE] - repmat(M$mean, length(idx), 1)
            dy_complete = self.Xt[self.complete_obs_idx, :]

            # Posterior mean of z: (n_latent, n_complete)
            # R line 28: x <- M$tau * Rxinv %*% t(M$PA) %*% t(dy)
            x = self.tau * Rx_inv @ self.weights.T @ dy_complete.T

            # Sufficient statistics
            # R line 29: T <- t(dy) %*% t(x)
            T = dy_complete.T @ x.T  # (n_var, n_latent)
            # R line 30: trS <- sum(sum(dy * dy))
            trS = np.sum(dy_complete**2)

            # Store scores for complete observations (R lines 33-36)
            scores[self.complete_obs_idx, :] = x.T
        else:
            T = np.zeros((self.n_var, self.n_latent))
            trS = 0.0

        # --- E-step for incomplete observations (R lines 39-60) ---
        if len(self.incomplete_obs_idx) > 0:
            for i in self.incomplete_obs_idx:
                # Mask for this observation
                is_missing = self.nan_mask[i, :]  # (n_var,)
                is_observed = ~is_missing
                n_missing = np.sum(is_missing)

                # Observed centered data for this row
                # R line 42: dyo <- y[i, !M$nans[i,], drop=FALSE] - M$mean[!M$nans[i,], drop=FALSE]
                dyo = self.Xt[i, is_observed]  # (n_observed,)

                # Partition weights
                # R lines 43-44
                Wm = self.weights[is_missing, :]  # (n_missing, n_latent)
                Wo = self.weights[is_observed, :]  # (n_observed, n_latent)

                # Modified precision matrix (remove contribution from missing features)
                # R line 45: Rxinv <- solve((Rx - M$tau * t(Wm) %*% Wm))
                Rx_inv_i = np.linalg.inv(Rx - self.tau * Wm.T @ Wm)

                # Posterior mean of z for this observation
                # R line 46: ex <- M$tau * t(Wo) %*% t(dyo)
                ex = self.tau * Wo.T @ dyo  # (n_latent,)
                # R line 47: x <- Rxinv %*% ex
                x = Rx_inv_i @ ex  # (n_latent,)

                # Impute missing values
                # R line 48: dym <- Wm %*% x
                dym = Wm @ x  # (n_missing,)

                # Reconstruct full dy vector
                # R lines 49-51
                dy = np.zeros(self.n_var)
                dy[is_observed] = dyo
                dy[is_missing] = dym

                # Update imputed data estimate
                # R line 52: M$yest[i,] <- dy + M$mean
                self.X_imputed[i, :] = dy + self.mu.flatten()

                # Accumulate sufficient statistics
                # R line 53: T <- T + t(dy) %*% t(x)
                T = T + np.outer(dy, x)

                # Special update for T at missing feature positions
                # R line 54: T[M$nans[i,], ] <- T[M$nans[i,],, drop=FALSE] + Wm %*% Rxinv
                T[is_missing, :] = T[is_missing, :] + Wm @ Rx_inv_i

                # Update trS
                # R lines 55-57: trS <- trS + dy %*% t(dy) + sum(M$nans[i,]) / M$tau +
                #                       sum(diag(Wm %*% Rxinv %*% t(Wm)))
                trS = trS + np.dot(dy, dy) + n_missing / self.tau + np.trace(Wm @ Rx_inv_i @ Wm.T)

                # Store scores (R line 59)
                scores[i, :] = x

        # Normalize by number of observations (R lines 62-63)
        T = T / self.n_obs
        trS = trS / self.n_obs

        return scores, T, trS, Rx

    def _m_step(self, T: np.ndarray, trS: float, Rx: np.ndarray) -> None:
        """Maximization step

        Finds parameters that maximize the expected loglikelihood.

        Parameters
        ----------
        scores
        T
            Cross covariance between latent variable and observed values
        trS

        Rx
            Precision matrix (n_latent, n_latent)
        """
        Rx_inv = np.linalg.inv(Rx)

        Dw = Rx_inv + self.tau * T.T @ self.weights @ Rx_inv + np.diag(self.alpha) / self.n_obs
        Dw_inv = np.linalg.inv(Dw)

        self.weights = T @ Dw_inv

        self.tau = float(self.n_var + 2 * self.GAMMA_TAU0 / self.n_obs) / (
            trS
            - np.trace(T.T @ self.weights)
            + (self.mu @ self.mu.T * self.GAMMA_MU0 + 2 * self.GAMMA_TAU0 / self.BETA_TAU0) / self.n_obs
        )

        self.var = Dw_inv * self.n_var / self.n_obs
        self.alpha = (2 * self.GAMMA_ALPHA0 + self.n_var) / (
            self.tau * np.diag(self.weights.T @ self.weights)
            + np.diag(self.var)
            + 2 * self.GAMMA_ALPHA0 / self.BETA_ALPHA0
        )

    def _convergence_criterium(self):
        """Convergence criterium"""

    @property
    def n_iter(self) -> int:
        """Number of iterations until convergence"""
        return self._n_iter

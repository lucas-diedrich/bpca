"""Core algorithm"""

import warnings

import numpy as np


class ConvergenceWarning(Warning):
    """Algorithm did not converge"""


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

    def __init__(
        self, X: np.ndarray, alpha: float = 1, sigma2: float = 1.0, max_iter: int = 1000, tol: float = 1e-6
    ) -> None:
        """Initialize Fit

        Parameters
        ----------
        X
            (n_var, n_obs)
        alpha
            ARD prior strength
        sigma2
            Variance prior
        max_iter
            Maximum number of EM iterations
        tol
            Convergence tolerance
        """
        self.X = X

        # Parameters
        self.alpha = alpha
        self.sigma2 = sigma2
        self._initialize_parameters()

        # Fitting parameters
        self.max_iter = max_iter
        self.tol = tol

        self._n_iter = None
        self._is_fit = False

    def _initialize_parameters(self):
        """Initialize fitting parameters

        Initializes w, z, mu, var
        """
        self.n_var, self.n_obs = self.X.shape
        self.n_latent = self.n_var - 1

        self.w = np.random.rand(self.n_var, self.n_latent)  # (n_features, n_latent)
        self.z = None
        self.mu = np.nanmean(self.X, axis=1, keepdims=True)  # (n_features,)
        self.var = self.sigma2 * np.eye(self.n_latent)  # (n_latent, n_latent)

        self.Xt = self.X - self.mu

        self.alpha = self.n_var / np.square(np.linalg.norm(self.w, axis=0))  # (n_latent,)

    def fit(self):
        """Fit model"""
        converged = False

        # Initialize M before loop (matches C reference)
        M = self.w.T @ self.w + self.sigma2 * np.eye(self.n_latent)

        for n_iter in range(self.max_iter):  # noqa: B007
            w_old = self.w.copy()
            sigma2_old = self.sigma2

            zn = self._e_step(M)
            M = self._m_step(zn=zn, M=M)

            # Convergence check: min of W change and sigma2 change
            w_change = np.linalg.norm(self.w - w_old)
            sigma2_change = np.abs(self.sigma2 - sigma2_old)
            tolgap = min(w_change, sigma2_change)

            if tolgap < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(f"Algorithm did not converge after {self.max_iter} steps", ConvergenceWarning, stacklevel=2)

        # Store final latent representation
        self.z = zn

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
        # z = M^{-1} @ W^T @ Xt, computed via solve for numerical stability
        z = np.linalg.solve(M, self.w.T @ self.Xt)

        return z

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
        # Update W
        # W_LHS = X @ Z.T  (n_var, n_latent)
        w1 = self.Xt @ zn.T

        # W_RHS = σ² diag(α) + σ² N M + Z @ Z.T  (n_latent, n_latent)
        w2 = self.sigma2 * np.diag(self.alpha) + self.sigma2 * self.n_obs * M + zn @ zn.T
        self.w = np.linalg.solve(w2.T, w1.T).T  # Equivalent to w1 @ inv(w2)

        # Compute new M with OLD sigma2 (matches C reference timing)
        M_new = self.w.T @ self.w + self.sigma2 * np.eye(self.n_latent)

        # Update sigma
        # σ² = 1/(N*d) * Σ_i [||x_i||² - 2 z_i^T W^T x_i + tr((σ²M + z_i z_i^T) W^T W)]
        WtW = self.w.T @ self.w  # (n_latent, n_latent)

        s1 = np.sum(self.Xt**2, axis=0)  # (n_obs,)
        s2 = -2 * np.sum(zn * (self.w.T @ self.Xt), axis=0)  # (n_obs,)
        base_trace = self.sigma2 * np.trace(M @ WtW)  # Shared across observations
        per_obs_trace = np.sum(zn * (WtW @ zn), axis=0)  # z_i^T WtW z_i for each i
        s3 = base_trace + per_obs_trace  # (n_obs,)

        self.sigma2 = np.sum(s1 + s2 + s3) / (self.n_obs * self.n_var)

        # Update alpha (ARD hyperparameters)
        self.alpha = self.n_var / np.square(np.linalg.norm(self.w, axis=0))

        return M_new

    def _check_is_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("Run .fit() method first")

    @property
    def n_iter(self) -> int:
        """Number of iterations until convergence"""
        return self._n_iter

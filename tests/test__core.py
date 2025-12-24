"""Test core algorithm"""

import numpy as np
import pytest

from bpca._core import BPCAFit


class TestBPCAFitInit:
    """Test parameter initialization"""

    @pytest.fixture
    def array(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_obs = 4, n_var=3) array and feature-wise mean"""
        return np.arange(12).reshape(4, 3), np.array([[4.5, 5.5, 6.5]])

    @pytest.mark.parametrize(
        ("n_latent", "expected_n_latent"),
        [(None, 2), (2, 2), (4, 3), (10, 3)],
        ids=("None", "n_latent-limiting", "n_var-limiting", "larger-than-possible"),
    )
    def test_bpcafit_init_em_parameters(self, array: np.ndarray, n_latent: int | None, expected_n_latent: int) -> None:
        X, mean = array

        bpca = BPCAFit(X=X, n_latent=n_latent)

        # Initialization
        assert np.array_equal(bpca.X, X)

        # Computation of em parameters
        assert np.array_equal(bpca.mu, mean)
        assert bpca.n_latent == expected_n_latent

        assert np.array_equal(bpca.Xt, (X - mean))
        assert bpca.z is None
        assert bpca.weights.shape == (X.shape[1], expected_n_latent)
        assert (bpca.tau >= 1e-10) & (bpca.tau <= 1e10)
        assert bpca.alpha.shape == (expected_n_latent,)
        assert np.array_equal(bpca.var, np.eye(expected_n_latent))

    @pytest.mark.parametrize("tolerance", [1e-3, 0.1, 1])
    @pytest.mark.parametrize("max_iter", [1, 100, 1000])
    def test_bpcafit_init_fit_procedure_parameters(self, array, max_iter: int, tolerance: float) -> None:
        X, _ = array
        bpca = BPCAFit(X=X, max_iter=max_iter, tolerance=tolerance)
        # Initialization of fit procedure parameters
        assert bpca.max_iter == max_iter
        assert bpca.tolerance == tolerance

    def test_bpcafit_init_fit_results_parameters(self, array) -> None:
        X, _ = array
        bpca = BPCAFit(X=X)
        # Initialization of fit procedure parameters
        assert bpca._converged is None
        assert bpca._n_iter is None
        assert bpca._is_fit is False

    @pytest.mark.parametrize(
        ("X", "complete_mask", "missing_mask"),
        [(np.array([[0, 0, 0], [np.nan, 0, 0], [np.nan, 0, 0]]), np.array([0]), np.array([1, 2]))],
    )
    def test_bpcafit_init_complete(self, X: np.ndarray, complete_mask: np.ndarray, missing_mask: np.ndarray) -> None:
        bpca = BPCAFit(X=X)

        assert np.array_equal(bpca.complete_obs_idx, complete_mask)
        assert np.array_equal(bpca.incomplete_obs_idx, missing_mask)


class TestBPCAFitEstep:
    @pytest.fixture
    def array(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_obs = 4, n_var=3) array and feature-wise mean"""
        return np.arange(12).reshape(4, 3)

    def test_estep__return_values(self, array):
        """Assert shapes are correct"""
        bpca = BPCAFit(X=array, n_latent=2)
        scores, T, trs = bpca._e_step()

        assert scores.shape == (array.shape[0], 2)
        assert T.shape == (array.shape[1], 2)
        assert isinstance(trs, float)

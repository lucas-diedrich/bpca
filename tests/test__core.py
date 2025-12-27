"""Test core algorithm"""

import numpy as np
import pytest

from bpca._core import BPCAFit, ConvergenceWarning


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
        """(n_obs = 4, n_var=3) array"""
        return np.arange(12).reshape(4, 3)

    @pytest.fixture
    def array_with_missing(self) -> np.ndarray:
        """Array with a missing value"""
        arr = np.arange(12, dtype=float).reshape(4, 3)
        arr[1, 0] = np.nan
        return arr

    @pytest.fixture
    def array_many_missing(self) -> np.ndarray:
        """Array in which each observation has a missing value"""
        arr = np.arange(12, dtype=float).reshape(4, 3)
        arr[[0, 1, 2, 3], [0, 1, 2, 0]] = np.nan
        return arr

    @pytest.mark.parametrize("n_latent", [2])
    def test_estep__return_values(self, array: np.ndarray, n_latent: int) -> None:
        """Assert shapes are correct"""
        bpca = BPCAFit(X=array, n_latent=2)
        scores, T, trs, Rx = bpca._e_step()

        assert scores.shape == (array.shape[0], n_latent)
        assert T.shape == (array.shape[1], n_latent)
        assert isinstance(trs, float)
        assert Rx.shape == (n_latent, n_latent)

    @pytest.mark.parametrize("n_latent", [2])
    def test_estep__na_values(self, array_with_missing: np.ndarray, n_latent: int) -> None:
        """Assert shapes are correct"""
        bpca = BPCAFit(X=array_with_missing, n_latent=2)
        scores, T, trs, Rx = bpca._e_step()

        assert scores.shape == (array_with_missing.shape[0], n_latent)
        assert T.shape == (array_with_missing.shape[1], n_latent)
        assert isinstance(trs, float)
        assert Rx.shape == (n_latent, n_latent)

    @pytest.mark.parametrize("n_latent", [2])
    def test_estep__many_na_values(self, array_many_missing: np.ndarray, n_latent: int) -> None:
        """Assert shapes are correct"""
        bpca = BPCAFit(X=array_many_missing, n_latent=2)
        scores, T, trs, Rx = bpca._e_step()

        assert scores.shape == (array_many_missing.shape[0], n_latent)
        assert T.shape == (array_many_missing.shape[1], n_latent)
        assert isinstance(trs, float)
        assert Rx.shape == (n_latent, n_latent)


class TestBPCAFitMstep:
    @pytest.fixture
    def array(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_obs = 4, n_var=3) array and feature-wise mean"""
        return np.arange(12).reshape(4, 3)

    @pytest.mark.parametrize("n_latent", [2])
    def test_mstep__return_values(self, array: np.ndarray, n_latent: int) -> None:
        """Assert shapes are correct"""
        bpca = BPCAFit(X=array, n_latent=n_latent)
        _, T, trS, Rx = bpca._e_step()
        res = bpca._m_step(T=T, trS=trS, Rx=Rx)

        assert res is None


class TestBPCAFitFitStep:
    @pytest.fixture
    def array(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_obs = 4, n_var=3) array"""
        return np.arange(12).reshape(4, 3)

    @pytest.fixture
    def array_converge(self) -> tuple[np.ndarray, np.ndarray]:
        """(n_obs = 3, n_var=4) array"""
        return np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])

    @pytest.mark.parametrize("max_iter", [10])
    @pytest.mark.parametrize("n_latent", [2])
    def test_fit__return_values(self, array: np.ndarray, n_latent: int, max_iter: int) -> None:
        """Assert shapes are correct"""
        # Does not converge
        bpca = BPCAFit(X=array, n_latent=n_latent, max_iter=max_iter)

        assert bpca._is_fit is False

        bpca.fit()

        assert bpca._is_fit is True
        assert bpca.n_iter == max_iter
        assert bpca._converged is False

        assert bpca.z.shape == (array.shape[0], n_latent)
        assert bpca.weights.shape == (array.shape[1], n_latent)
        assert bpca.tau > 0
        assert bpca.alpha.shape == (n_latent,)
        assert (bpca.alpha > 0).all()

    def test_fit__convereges(self, array_converge: np.ndarray) -> None:
        """Assert shapes are correct"""
        # Does not converge
        bpca = BPCAFit(X=array_converge, n_latent=1, max_iter=100)

        assert bpca._is_fit is False

        bpca.fit()

        assert bpca._is_fit is True
        assert bpca._converged is True

    def test_fit_warns_on_non_convergence(self, array: np.ndarray) -> None:
        with pytest.warns(ConvergenceWarning):
            BPCAFit(X=array, n_latent=2, max_iter=1).fit()

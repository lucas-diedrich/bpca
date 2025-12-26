"""Test high-level API"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from bpca._bpca import BPCA, compute_variance_explained


class TestComputeVarianceExplained:
    @pytest.fixture(params=[{"random_state": random_state} for random_state in range(50)])
    def random_array(self, request) -> np.ndarray:
        random_state = request.param["random_state"]
        rng = np.random.default_rng(seed=random_state)

        return rng.normal(loc=0, scale=1, size=(50, 100))

    @pytest.fixture
    def array_with_nan(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.array([[-1, np.nan], [0, 0], [1, 0]])

        usage = np.array([[-1, 0, 1]]).T
        loadings = np.array([[1, 0]])

        explained_variance = np.array([1.0])

        return X, usage, loadings, explained_variance

    def test_compute_variance_explained(self, random_array: np.ndarray) -> None:
        """Test that variance explained computation yields consistent results with default procedure for PCA"""
        pca = PCA(n_components=10)
        usage = pca.fit_transform(random_array)
        loadings = pca.components_

        res = compute_variance_explained(random_array, usage, loadings)

        assert np.allclose(pca.explained_variance_ratio_, res)

    def test_compute_variance_explained_supports_nan(self, array_with_nan: tuple[np.ndarray]):
        X, usage, loadings, explained_variance = array_with_nan
        result = compute_variance_explained(X, usage, loadings)
        assert np.allclose(result, explained_variance)


class TestBPCAInit:
    """Test BPCA initialization"""

    @pytest.mark.parametrize(
        ("n_components", "max_iter", "tolerance"),
        [(None, 1000, 1e-4), (5, 500, 1e-3), (10, 100, 0.1)],
        ids=("default", "custom-1", "custom-2"),
    )
    def test_init_stores_parameters(self, n_components: int | None, max_iter: int, tolerance: float) -> None:
        bpca = BPCA(n_components=n_components, max_iter=max_iter, tolerance=tolerance)

        assert bpca._n_components == n_components
        assert bpca._max_iter == max_iter
        assert bpca._tolerance == tolerance


class TestBPCAFit:
    """Test BPCA fit method"""

    @pytest.fixture
    def array(self) -> np.ndarray:
        """(n_obs=20, n_var=10) array"""
        rng = np.random.default_rng(seed=42)
        return rng.normal(size=(20, 10))

    @pytest.fixture
    def array_with_missing(self) -> np.ndarray:
        """Array with missing values"""
        rng = np.random.default_rng(seed=42)
        arr = rng.normal(size=(20, 10))
        arr[0, 0] = np.nan
        arr[5, 3] = np.nan
        return arr

    def test_fit_returns_self(self, array: np.ndarray) -> None:
        bpca = BPCA(n_components=3, max_iter=10)

        result = bpca.fit(array)

        assert result is bpca

    @pytest.mark.parametrize("n_components", [2, 5])
    def test_fit_sets_attributes(self, array: np.ndarray, n_components: int) -> None:
        bpca = BPCA(n_components=n_components, max_iter=50)

        bpca.fit(array)

        assert bpca._is_fit is True
        assert bpca._mu.shape == (1, array.shape[1])
        assert bpca._usage.shape == (array.shape[0], n_components)
        assert bpca._components.shape == (n_components, array.shape[1])
        assert bpca._alpha.shape == (n_components,)
        assert isinstance(bpca._tau, float)
        assert bpca._tau > 0

    def test_fit_with_missing_values(self, array_with_missing: np.ndarray) -> None:
        bpca = BPCA(n_components=3, max_iter=50)

        bpca.fit(array_with_missing)

        assert bpca._is_fit is True


class TestBPCATransform:
    """Test BPCA transform method"""

    @pytest.fixture
    def fitted_bpca(self) -> tuple[BPCA, np.ndarray]:
        """Fitted BPCA model and training data"""
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=(20, 10))
        bpca = BPCA(n_components=3, max_iter=50)
        bpca.fit(X)
        return bpca, X

    def test_transform_raises_if_not_fit(self) -> None:
        bpca = BPCA(n_components=3)
        X = np.random.default_rng(0).normal(size=(10, 5))

        with pytest.raises(RuntimeError):
            bpca.transform(X)

    def test_transform_returns_correct_shape(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca
        X_new = np.random.default_rng(0).normal(size=(5, 10))

        result = bpca.transform(X_new)

        assert result.shape == (5, 3)


class TestBPCAFitTransform:
    """Test BPCA transform method"""

    @pytest.fixture
    def array(self) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        usage = rng.normal(size=(20, 5))
        loadings = rng.normal(size=(5, 10))

        return usage @ loadings

    def test_bpca_fit_transform(self, array: np.ndarray) -> None:
        """Fitted BPCA model and training data"""
        bpca = BPCA(n_components=3, max_iter=50)

        assert bpca._is_fit is False

        result = bpca.fit_transform(array)

        assert result.shape == (20, 3)
        assert bpca._is_fit is True


class TestBPCAProperties:
    """Test BPCA properties"""

    @pytest.fixture
    def fitted_bpca(self) -> tuple[BPCA, np.ndarray]:
        """Fitted BPCA model and training data"""
        rng = np.random.default_rng(seed=42)
        usage = rng.normal(size=(20, 3))
        loadings = rng.normal(size=(3, 10))
        X = usage @ loadings

        bpca = BPCA(n_components=3, max_iter=50)
        bpca.fit(X)
        return bpca, X

    @pytest.fixture
    def unfitted_bpca(self) -> BPCA:
        """Unfitted BPCA model"""
        return BPCA(n_components=3)

    @pytest.mark.parametrize(
        "property_name",
        ["components_", "explained_variance_ratio_", "n_iter", "alpha", "tau"],
    )
    def test_properties_raise_if_not_fit(self, unfitted_bpca: BPCA, property_name: str) -> None:
        with pytest.raises(RuntimeError, match="Fit model first"):
            getattr(unfitted_bpca, property_name)

    def test_components_shape(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, X = fitted_bpca

        result = bpca.components_

        assert result.shape == (3, X.shape[1])

    def test_explained_variance_ratio_shape(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        result = bpca.explained_variance_ratio_

        assert result.shape == (3,)
        assert (result >= 0).all()

    def test_n_components_returns_value(self) -> None:
        bpca = BPCA(n_components=5)

        assert bpca.n_components_ == 5

    def test_n_iter_returns_positive(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        assert bpca.n_iter > 0

    def test_alpha_shape_and_sorted(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        result = bpca.alpha

        assert result.shape == (3,)
        assert np.all(result[:-1] <= result[1:])  # sorted ascending

    def test_tau_positive(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        assert bpca.tau > 0

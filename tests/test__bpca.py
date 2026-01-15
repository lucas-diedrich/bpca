"""Test high-level API"""

import numpy as np
import pytest

from bpca._bpca import BPCA


class TestBPCAInit:
    """Test BPCA initialization"""

    @pytest.mark.parametrize("n_components", [None, 2], ids=("default", "non-default-2"))
    @pytest.mark.parametrize("max_iter", [1000, 500], ids=("default", "non-default-500"))
    @pytest.mark.parametrize("tolerance", [1e-4, 0.1, 1e-6], ids=("default", "non-default-higher", "non-default-lower"))
    @pytest.mark.parametrize("sort_components", [True, False], ids=("sort-true", "sort-false"))
    def test_init_stores_parameters(
        self, n_components: int | None, max_iter: int, tolerance: float, sort_components: bool
    ) -> None:
        bpca = BPCA(n_components=n_components, max_iter=max_iter, tolerance=tolerance, sort_components=sort_components)

        assert bpca._n_components == n_components
        assert bpca._max_iter == max_iter
        assert bpca._tolerance == tolerance
        assert bpca._sort_components == sort_components


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
        ["components_", "explained_variance_ratio_", "n_iter", "alpha", "tau", "mu"],
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

    def test_alpha_shape(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        result = bpca.alpha

        assert result.shape == (3,)

    def test_tau_positive(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        bpca, _ = fitted_bpca

        assert bpca.tau > 0

    def test_mu_shape(self, fitted_bpca: tuple[BPCA, np.ndarray]) -> None:
        """Test that the mu parameter is accessible and has the expected shape"""
        bpca, X = fitted_bpca

        assert bpca.mu.shape == (X.shape[1],)


class TestBPCASortComponents:
    """Test BPCA sort_components parameter"""

    @pytest.fixture
    def array(self) -> np.ndarray:
        """(n_obs=50, n_var=10) array with latent structure"""
        rng = np.random.default_rng(seed=42)
        usage = rng.normal(size=(50, 5))
        loadings = rng.normal(size=(5, 10))
        return usage @ loadings

    def test_sort_components_false_preserves_order(self, array: np.ndarray) -> None:
        """Default behavior: components not sorted by explained variance."""
        bpca = BPCA(n_components=3, sort_components=False, max_iter=100)
        bpca.fit(array)

        assert bpca.components_.shape == (3, 10)
        assert bpca.explained_variance_ratio_.shape == (3,)

    def test_sort_components_true_sorts_by_variance(self, array: np.ndarray) -> None:
        """When sort_components=True, components sorted by explained variance descending."""
        bpca = BPCA(n_components=3, sort_components=True, max_iter=100)
        bpca.fit(array)

        evr = bpca.explained_variance_ratio_
        assert np.all(evr[:-1] >= evr[1:]), "Explained variance should be in descending order"

    def test_sort_components_reconstruction_invariant(self, array: np.ndarray) -> None:
        """Sorting should not affect reconstruction quality."""
        bpca_unsorted = BPCA(n_components=3, sort_components=False, max_iter=100)
        bpca_sorted = BPCA(n_components=3, sort_components=True, max_iter=100)

        usage_unsorted = bpca_unsorted.fit_transform(array)
        usage_sorted = bpca_sorted.fit_transform(array)

        recon_unsorted = usage_unsorted @ bpca_unsorted.components_
        recon_sorted = usage_sorted @ bpca_sorted.components_

        np.testing.assert_allclose(recon_unsorted, recon_sorted, atol=1e-10)

"""Test utilities module"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from bpca._utils import compute_variance_explained


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

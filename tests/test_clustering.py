"""
tests/test_clustering.py
=========================
Testes para os algoritmos de clustering e o módulo de avaliação
(ml_clustering_lab.clustering).

Cobertura
---------
- ClusteringBase: interface (name, supports_noise, requires_k, fit_predict, get_params)
- KMeansRunner: instanciação, get_params, fit_predict
- DBSCANRunner: instanciação, get_params, fit_predict, supports_noise=True
- AgglomerativeRunner: instanciação, get_params, fit_predict
- MeanShiftRunner: instanciação, get_params, fit_predict
- get_algorithm: resolução pelo nome e erro para nome inválido
- evaluation.compute_internal_metrics
- evaluation.compute_external_metrics
- evaluation.build_comparison_table
"""

import numpy as np
import pytest

from ml_clustering_lab.clustering import (
    AgglomerativeRunner,
    DBSCANRunner,
    KMeansRunner,
    MeanShiftRunner,
    get_algorithm,
)
from ml_clustering_lab.clustering.evaluation import (
    build_comparison_table,
    compute_external_metrics,
    compute_internal_metrics,
)


@pytest.fixture
def simple_X() -> np.ndarray:
    """Array 2D com 3 clusters bem separados para testes."""
    rng = np.random.default_rng(42)
    c1 = rng.standard_normal((30, 2)) + np.array([0, 0])
    c2 = rng.standard_normal((30, 2)) + np.array([8, 0])
    c3 = rng.standard_normal((30, 2)) + np.array([4, 8])
    return np.vstack([c1, c2, c3])


@pytest.fixture
def simple_labels(simple_X) -> np.ndarray:
    """Labels verdadeiros correspondentes ao simple_X."""
    return np.array([0] * 30 + [1] * 30 + [2] * 30)


class TestGetAlgorithm:
    def test_kmeans(self):
        algo = get_algorithm("kmeans", n_clusters=3)
        assert isinstance(algo, KMeansRunner)

    def test_dbscan(self):
        algo = get_algorithm("dbscan")
        assert isinstance(algo, DBSCANRunner)

    def test_agglomerative(self):
        algo = get_algorithm("agglomerative")
        assert isinstance(algo, AgglomerativeRunner)

    def test_mean_shift(self):
        algo = get_algorithm("mean-shift")
        assert isinstance(algo, MeanShiftRunner)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="não encontrado"):
            get_algorithm("algoritmo_inexistente")


class TestKMeansRunner:
    def test_instantiation_defaults(self):
        runner = KMeansRunner()
        assert runner.n_clusters == 3
        assert runner.init == "k-means++"
        assert runner.requires_k is True
        assert runner.supports_noise is False

    def test_get_params(self):
        runner = KMeansRunner(n_clusters=5)
        params = runner.get_params()
        assert params["n_clusters"] == 5
        assert "init" in params
        assert "max_iter" in params

    def test_fit_predict_shape(self, simple_X):
        runner = KMeansRunner(n_clusters=3)
        labels = runner.fit_predict(simple_X)
        assert labels.shape == (simple_X.shape[0],)

    def test_fit_predict_n_clusters(self, simple_X):
        runner = KMeansRunner(n_clusters=3)
        labels = runner.fit_predict(simple_X)
        assert len(set(labels)) == 3

    def test_fit_predict_dtype(self, simple_X):
        runner = KMeansRunner(n_clusters=3)
        labels = runner.fit_predict(simple_X)
        assert np.issubdtype(labels.dtype, np.integer)


class TestDBSCANRunner:
    def test_instantiation_defaults(self):
        runner = DBSCANRunner()
        assert runner.eps == 0.5
        assert runner.min_samples == 5
        assert runner.supports_noise is True
        assert runner.requires_k is False

    def test_get_params(self):
        runner = DBSCANRunner(eps=0.3, min_samples=10)
        params = runner.get_params()
        assert params["eps"] == 0.3
        assert params["min_samples"] == 10

    def test_fit_predict_shape(self, simple_X):
        runner = DBSCANRunner(eps=1.5, min_samples=5)
        labels = runner.fit_predict(simple_X)
        assert labels.shape == (simple_X.shape[0],)

    def test_fit_predict_finds_clusters(self, simple_X):
        runner = DBSCANRunner(eps=1.5, min_samples=5)
        labels = runner.fit_predict(simple_X)
        # Should find at least 1 cluster (label != -1)
        assert (labels != -1).any()


class TestAgglomerativeRunner:
    def test_instantiation_defaults(self):
        runner = AgglomerativeRunner()
        assert runner.n_clusters == 3
        assert runner.linkage == "ward"
        assert runner.requires_k is True
        assert runner.supports_noise is False

    def test_get_params(self):
        runner = AgglomerativeRunner(n_clusters=4, linkage="complete")
        params = runner.get_params()
        assert params["n_clusters"] == 4
        assert params["linkage"] == "complete"

    def test_fit_predict_shape(self, simple_X):
        runner = AgglomerativeRunner(n_clusters=3)
        labels = runner.fit_predict(simple_X)
        assert labels.shape == (simple_X.shape[0],)

    def test_fit_predict_n_clusters(self, simple_X):
        runner = AgglomerativeRunner(n_clusters=3)
        labels = runner.fit_predict(simple_X)
        assert len(set(labels)) == 3


class TestMeanShiftRunner:
    def test_instantiation_defaults(self):
        runner = MeanShiftRunner()
        assert runner.bandwidth is None
        assert runner.requires_k is False
        assert runner.supports_noise is False

    def test_get_params(self):
        runner = MeanShiftRunner(bandwidth=1.5)
        params = runner.get_params()
        assert params["bandwidth"] == 1.5

    def test_fit_predict_shape(self, simple_X):
        runner = MeanShiftRunner()
        labels = runner.fit_predict(simple_X)
        assert labels.shape == (simple_X.shape[0],)

    def test_fit_predict_finds_clusters(self, simple_X):
        runner = MeanShiftRunner()
        labels = runner.fit_predict(simple_X)
        assert len(set(labels)) >= 1


class TestComputeInternalMetrics:
    def test_returns_expected_keys(self, simple_X):
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        metrics = compute_internal_metrics(simple_X, labels)
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "calinski_harabasz" in metrics
        assert "n_clusters" in metrics
        assert "n_noise" in metrics

    def test_n_clusters_count(self, simple_X):
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        metrics = compute_internal_metrics(simple_X, labels)
        assert metrics["n_clusters"] == 3

    def test_noise_count_dbscan(self, simple_X):
        labels = np.array([-1] * 5 + [0] * 40 + [1] * 45)
        metrics = compute_internal_metrics(simple_X, labels)
        assert metrics["n_noise"] == 5

    def test_silhouette_range(self, simple_X):
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        metrics = compute_internal_metrics(simple_X, labels)
        assert -1.0 <= metrics["silhouette"] <= 1.0


class TestComputeExternalMetrics:
    def test_perfect_match(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        metrics = compute_external_metrics(labels, labels)
        assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
        assert metrics["normalized_mutual_info"] == pytest.approx(1.0)
        assert metrics["v_measure"] == pytest.approx(1.0)

    def test_returns_expected_keys(self):
        labels = np.array([0, 0, 1, 1])
        metrics = compute_external_metrics(labels, labels)
        for key in ["adjusted_rand_index", "normalized_mutual_info", "homogeneity", "completeness", "v_measure"]:
            assert key in metrics


class TestBuildComparisonTable:
    def test_returns_dataframe(self):
        import pandas as pd
        results = [
            {"algorithm": "K-Means", "n_clusters": 3.0, "n_noise": 0.0, "silhouette": 0.55, "davies_bouldin": 0.80, "calinski_harabasz": 200.0, "elapsed_time": 0.1},
            {"algorithm": "DBSCAN", "n_clusters": 2.0, "n_noise": 5.0, "silhouette": 0.40, "davies_bouldin": 1.10, "calinski_harabasz": 150.0, "elapsed_time": 0.05},
        ]
        table = build_comparison_table(results)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2

    def test_sorted_by_silhouette(self):
        results = [
            {"algorithm": "B", "silhouette": 0.3},
            {"algorithm": "A", "silhouette": 0.7},
        ]
        table = build_comparison_table(results)
        assert table.iloc[0]["algorithm"] == "A"

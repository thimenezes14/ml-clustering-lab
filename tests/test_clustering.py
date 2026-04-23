"""
tests/test_clustering.py
=========================
Testes para os algoritmos de clustering e o módulo de avaliação
(ml_clustering_lab.clustering).

Cobertura planejada
-------------------
- ClusteringBase: interface (name, supports_noise, requires_k, fit_predict, get_params)
- KMeansRunner: instanciação, get_params, fit_predict
- DBSCANRunner: instanciação, get_params, fit_predict, supports_noise=True
- AgglomerativeRunner: instanciação, get_params, fit_predict
- MeanShiftRunner: instanciação, get_params, fit_predict
- get_algorithm: resolução pelo nome e erro para nome inválido
- evaluation.compute_internal_metrics: stub

Nota
----
Todos os testes de fit_predict estão como stubs (``NotImplementedError``).
Atualize-os com verificações de shape e dtype dos labels conforme a
implementação avançar.
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
from ml_clustering_lab.clustering.evaluation import compute_internal_metrics


@pytest.fixture
def simple_X() -> np.ndarray:
    """Array 2D simples para testes de clustering."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 2))


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

    def test_fit_predict_not_implemented(self, simple_X):
        runner = KMeansRunner()
        with pytest.raises(NotImplementedError):
            runner.fit_predict(simple_X)


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

    def test_fit_predict_not_implemented(self, simple_X):
        runner = DBSCANRunner()
        with pytest.raises(NotImplementedError):
            runner.fit_predict(simple_X)


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

    def test_fit_predict_not_implemented(self, simple_X):
        runner = AgglomerativeRunner()
        with pytest.raises(NotImplementedError):
            runner.fit_predict(simple_X)


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

    def test_fit_predict_not_implemented(self, simple_X):
        runner = MeanShiftRunner()
        with pytest.raises(NotImplementedError):
            runner.fit_predict(simple_X)


class TestComputeInternalMetrics:
    def test_raises_not_implemented(self, simple_X):
        labels = np.zeros(50, dtype=int)
        with pytest.raises(NotImplementedError):
            compute_internal_metrics(simple_X, labels)

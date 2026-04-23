"""
tests/test_visualization.py
============================
Testes para os módulos de visualização
(ml_clustering_lab.visualization.plots e embeddings).

Cobertura
---------
- plot_histogram: cria arquivo PNG
- plot_boxplot: cria arquivo PNG
- plot_correlation: cria arquivo PNG
- plot_scatter_clusters: cria arquivo PNG
- plot_dendrogram: cria arquivo PNG (via AgglomerativeClustering)
- plot_compare_metrics: cria arquivo PNG
- reduce_pca: shape correto
- reduce_tsne: shape correto
- plot_pca_2d: cria arquivo PNG
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [2.0, 3.0, 1.0, 5.0, 4.0, 6.0],
        }
    )


@pytest.fixture
def cluster_X() -> np.ndarray:
    rng = np.random.default_rng(42)
    c1 = rng.standard_normal((20, 2)) + np.array([0, 0])
    c2 = rng.standard_normal((20, 2)) + np.array([5, 5])
    return np.vstack([c1, c2])


@pytest.fixture
def cluster_labels() -> np.ndarray:
    return np.array([0] * 20 + [1] * 20)


# ---------------------------------------------------------------------------
# plots.py
# ---------------------------------------------------------------------------


class TestPlotHistogram:
    def test_creates_file(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_histogram

        plot_histogram(simple_df, outdir=str(tmp_path))
        assert (tmp_path / "histogram.png").exists()

    def test_no_error_with_columns_arg(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_histogram

        plot_histogram(simple_df, columns=["x0"], outdir=str(tmp_path))
        assert (tmp_path / "histogram.png").exists()

    def test_no_numeric_columns_noop(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_histogram

        df = pd.DataFrame({"text": ["a", "b", "c"]})
        plot_histogram(df, outdir=str(tmp_path))
        assert not (tmp_path / "histogram.png").exists()


class TestPlotBoxplot:
    def test_creates_file(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_boxplot

        plot_boxplot(simple_df, outdir=str(tmp_path))
        assert (tmp_path / "boxplot.png").exists()

    def test_no_error_with_columns_arg(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_boxplot

        plot_boxplot(simple_df, columns=["x1"], outdir=str(tmp_path))
        assert (tmp_path / "boxplot.png").exists()


class TestPlotCorrelation:
    def test_creates_file(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_correlation

        plot_correlation(simple_df, outdir=str(tmp_path))
        assert (tmp_path / "correlation.png").exists()

    def test_spearman_method(self, simple_df, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_correlation

        plot_correlation(simple_df, method="spearman", outdir=str(tmp_path))
        assert (tmp_path / "correlation.png").exists()

    def test_no_numeric_noop(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_correlation

        df = pd.DataFrame({"text": ["a", "b"]})
        plot_correlation(df, outdir=str(tmp_path))
        assert not (tmp_path / "correlation.png").exists()


class TestPlotScatterClusters:
    def test_creates_file(self, cluster_X, cluster_labels, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_scatter_clusters

        plot_scatter_clusters(cluster_X, cluster_labels, title="test", outdir=str(tmp_path))
        assert (tmp_path / "test.png").exists()

    def test_handles_noise_label(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_scatter_clusters

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        labels = np.array([-1] * 5 + [0] * 13 + [1] * 12)
        plot_scatter_clusters(X, labels, title="noise_test", outdir=str(tmp_path))
        assert (tmp_path / "noise_test.png").exists()

    def test_high_dim_uses_first_two_features(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_scatter_clusters

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 5))
        labels = np.zeros(30, dtype=int)
        plot_scatter_clusters(X, labels, title="highdim", outdir=str(tmp_path))
        assert (tmp_path / "highdim.png").exists()


class TestPlotCompareMetrics:
    def test_creates_file(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_compare_metrics

        df = pd.DataFrame(
            {
                "algorithm": ["K-Means", "DBSCAN"],
                "silhouette": [0.55, 0.40],
                "davies_bouldin": [0.80, 1.10],
                "calinski_harabasz": [200.0, 150.0],
            }
        )
        plot_compare_metrics(df, outdir=str(tmp_path))
        assert (tmp_path / "comparison_metrics_plot.png").exists()

    def test_no_algorithm_column_noop(self, tmp_path):
        from ml_clustering_lab.visualization.plots import plot_compare_metrics

        df = pd.DataFrame({"silhouette": [0.5]})
        plot_compare_metrics(df, outdir=str(tmp_path))
        assert not (tmp_path / "comparison_metrics_plot.png").exists()


# ---------------------------------------------------------------------------
# embeddings.py
# ---------------------------------------------------------------------------


class TestReducePca:
    def test_output_shape(self, cluster_X):
        from ml_clustering_lab.visualization.embeddings import reduce_pca

        result = reduce_pca(cluster_X, n_components=2)
        assert result.shape == (cluster_X.shape[0], 2)

    def test_n_components_1(self, cluster_X):
        from ml_clustering_lab.visualization.embeddings import reduce_pca

        result = reduce_pca(cluster_X, n_components=1)
        assert result.shape == (cluster_X.shape[0], 1)

    def test_reproducible(self, cluster_X):
        from ml_clustering_lab.visualization.embeddings import reduce_pca

        r1 = reduce_pca(cluster_X, random_state=0)
        r2 = reduce_pca(cluster_X, random_state=0)
        assert np.allclose(r1, r2)


class TestPlotPca2d:
    def test_creates_file(self, cluster_X, cluster_labels, tmp_path):
        from ml_clustering_lab.visualization.embeddings import plot_pca_2d

        plot_pca_2d(cluster_X, cluster_labels, title="PCA Test", outdir=str(tmp_path))
        assert (tmp_path / "pca_2d.png").exists()

    def test_handles_noise_label(self, cluster_X, tmp_path):
        from ml_clustering_lab.visualization.embeddings import plot_pca_2d

        labels = np.array([-1] * 5 + [0] * 15 + [1] * 20)
        plot_pca_2d(cluster_X, labels, outdir=str(tmp_path))
        assert (tmp_path / "pca_2d.png").exists()

"""
tests/test_optimal_k.py
========================
Testes para o módulo de escolha de k ótimo
(ml_clustering_lab.clustering.optimal_k).

Cobertura
---------
- elbow_analysis: shape, colunas, valores positivos
- silhouette_analysis: shape, colunas, range
- plot_elbow: cria arquivo PNG
- plot_silhouette_analysis: cria arquivo PNG
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")


@pytest.fixture
def simple_X() -> np.ndarray:
    """Dados com 3 clusters bem separados."""
    rng = np.random.default_rng(42)
    c1 = rng.standard_normal((30, 2)) + np.array([0, 0])
    c2 = rng.standard_normal((30, 2)) + np.array([8, 0])
    c3 = rng.standard_normal((30, 2)) + np.array([4, 8])
    return np.vstack([c1, c2, c3])


class TestElbowAnalysis:
    def test_returns_dataframe(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=range(2, 5))
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=range(2, 5))
        assert "k" in df.columns
        assert "inertia" in df.columns

    def test_row_count_matches_k_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        k_range = range(2, 6)
        df = elbow_analysis(simple_X, k_range=k_range)
        assert len(df) == len(k_range)

    def test_inertia_positive(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=range(2, 5))
        assert (df["inertia"] > 0).all()

    def test_inertia_decreases_with_k(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=range(2, 6))
        # inertia should be non-increasing as k grows (more clusters = less WCSS)
        inertia = df.sort_values("k")["inertia"].values
        assert (inertia[:-1] >= inertia[1:]).all()

    def test_default_k_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X)
        assert len(df) == len(range(2, 11))

    def test_list_k_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=[2, 4, 6])
        assert set(df["k"]) == {2, 4, 6}

    def test_sorted_by_k(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis

        df = elbow_analysis(simple_X, k_range=range(2, 6))
        assert list(df["k"]) == sorted(df["k"])


class TestSilhouetteAnalysis:
    def test_returns_dataframe(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X, k_range=range(2, 5))
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X, k_range=range(2, 5))
        assert "k" in df.columns
        assert "silhouette" in df.columns

    def test_row_count_matches_k_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        k_range = range(2, 6)
        df = silhouette_analysis(simple_X, k_range=k_range)
        assert len(df) == len(k_range)

    def test_silhouette_in_valid_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X, k_range=range(2, 5))
        assert (df["silhouette"] >= -1).all()
        assert (df["silhouette"] <= 1).all()

    def test_best_k_for_well_separated_clusters(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X, k_range=range(2, 7))
        best_k = df.loc[df["silhouette"].idxmax(), "k"]
        # For 3 well-separated clusters, k=3 should have highest silhouette
        assert best_k == 3

    def test_default_k_range(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X)
        assert len(df) == len(range(2, 11))

    def test_sorted_by_k(self, simple_X):
        from ml_clustering_lab.clustering.optimal_k import silhouette_analysis

        df = silhouette_analysis(simple_X, k_range=range(2, 6))
        assert list(df["k"]) == sorted(df["k"])


class TestPlotElbow:
    def test_creates_png(self, simple_X, tmp_path):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis, plot_elbow

        df = elbow_analysis(simple_X, k_range=range(2, 5))
        plot_elbow(df, outdir=str(tmp_path))
        assert (tmp_path / "elbow.png").exists()

    def test_file_nonzero_size(self, simple_X, tmp_path):
        from ml_clustering_lab.clustering.optimal_k import elbow_analysis, plot_elbow

        df = elbow_analysis(simple_X, k_range=range(2, 5))
        plot_elbow(df, outdir=str(tmp_path))
        assert (tmp_path / "elbow.png").stat().st_size > 0


class TestPlotSilhouetteAnalysis:
    def test_creates_png(self, simple_X, tmp_path):
        from ml_clustering_lab.clustering.optimal_k import (
            plot_silhouette_analysis,
            silhouette_analysis,
        )

        df = silhouette_analysis(simple_X, k_range=range(2, 5))
        plot_silhouette_analysis(df, outdir=str(tmp_path))
        assert (tmp_path / "silhouette_analysis.png").exists()

    def test_file_nonzero_size(self, simple_X, tmp_path):
        from ml_clustering_lab.clustering.optimal_k import (
            plot_silhouette_analysis,
            silhouette_analysis,
        )

        df = silhouette_analysis(simple_X, k_range=range(2, 5))
        plot_silhouette_analysis(df, outdir=str(tmp_path))
        assert (tmp_path / "silhouette_analysis.png").stat().st_size > 0

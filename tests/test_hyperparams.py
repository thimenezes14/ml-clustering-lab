"""
tests/test_hyperparams.py
==========================
Testes para o módulo de seleção de hiperparâmetros para DBSCAN e Mean Shift
(ml_clustering_lab.clustering.hyperparams).

Cobertura
---------
- k_distance_analysis: shape, colunas, valores positivos, ordenação
- plot_k_distance: cria arquivo PNG
- estimate_bandwidth_range: shape, colunas, valores positivos, ordenação
- plot_bandwidth_range: cria arquivo PNG
- Exportação via clustering.__init__
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")


@pytest.fixture
def blobs_X() -> np.ndarray:
    """Dados com 3 clusters bem separados (escalonados)."""
    rng = np.random.default_rng(0)
    c1 = rng.standard_normal((40, 2)) + np.array([0.0, 0.0])
    c2 = rng.standard_normal((40, 2)) + np.array([6.0, 0.0])
    c3 = rng.standard_normal((40, 2)) + np.array([3.0, 6.0])
    return np.vstack([c1, c2, c3])


# ---------------------------------------------------------------------------
# k_distance_analysis
# ---------------------------------------------------------------------------


class TestKDistanceAnalysis:
    def test_returns_dataframe(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        assert "sample_rank" in df.columns
        assert "k_distance" in df.columns

    def test_row_count_equals_n_samples(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        assert len(df) == len(blobs_X)

    def test_distances_are_positive(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        assert (df["k_distance"] >= 0).all()

    def test_distances_sorted_ascending(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        vals = df["k_distance"].values
        assert (vals[:-1] <= vals[1:]).all()

    def test_sample_rank_is_sequential(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=5)
        assert list(df["sample_rank"]) == list(range(len(blobs_X)))

    def test_different_k_changes_distances(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis

        df3 = k_distance_analysis(blobs_X, k=3)
        df10 = k_distance_analysis(blobs_X, k=10)
        # larger k → larger or equal k-distances
        assert df10["k_distance"].mean() >= df3["k_distance"].mean()

    def test_exported_from_init(self, blobs_X):
        from ml_clustering_lab.clustering import k_distance_analysis

        df = k_distance_analysis(blobs_X, k=4)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# plot_k_distance
# ---------------------------------------------------------------------------


class TestPlotKDistance:
    def test_creates_png(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis, plot_k_distance

        df = k_distance_analysis(blobs_X, k=5)
        plot_k_distance(df, outdir=str(tmp_path))
        assert (tmp_path / "k_distance.png").exists()

    def test_file_nonzero_size(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering.hyperparams import k_distance_analysis, plot_k_distance

        df = k_distance_analysis(blobs_X, k=5)
        plot_k_distance(df, outdir=str(tmp_path))
        assert (tmp_path / "k_distance.png").stat().st_size > 0

    def test_exported_from_init(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering import k_distance_analysis, plot_k_distance

        df = k_distance_analysis(blobs_X, k=5)
        plot_k_distance(df, outdir=str(tmp_path))
        assert (tmp_path / "k_distance.png").exists()


# ---------------------------------------------------------------------------
# estimate_bandwidth_range
# ---------------------------------------------------------------------------


class TestEstimateBandwidthRange:
    def test_returns_dataframe(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X)
        assert "quantile" in df.columns
        assert "bandwidth" in df.columns

    def test_default_has_nine_rows(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X)
        assert len(df) == 9

    def test_custom_quantile_range(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.2, 0.5, 0.8])
        assert len(df) == 3
        assert set(df["quantile"]) == {0.2, 0.5, 0.8}

    def test_bandwidth_positive(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X)
        assert (df["bandwidth"] > 0).all()

    def test_bandwidth_increases_with_quantile(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X)
        bw = df.sort_values("quantile")["bandwidth"].values
        # bandwidth should be non-decreasing as quantile grows
        assert (bw[:-1] <= bw[1:]).all()

    def test_sorted_by_quantile(self, blobs_X):
        from ml_clustering_lab.clustering.hyperparams import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.5, 0.1, 0.3])
        assert list(df["quantile"]) == sorted(df["quantile"])

    def test_exported_from_init(self, blobs_X):
        from ml_clustering_lab.clustering import estimate_bandwidth_range

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.3, 0.5])
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# plot_bandwidth_range
# ---------------------------------------------------------------------------


class TestPlotBandwidthRange:
    def test_creates_png(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering.hyperparams import (
            estimate_bandwidth_range,
            plot_bandwidth_range,
        )

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.2, 0.5, 0.8])
        plot_bandwidth_range(df, outdir=str(tmp_path))
        assert (tmp_path / "bandwidth_range.png").exists()

    def test_file_nonzero_size(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering.hyperparams import (
            estimate_bandwidth_range,
            plot_bandwidth_range,
        )

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.2, 0.5, 0.8])
        plot_bandwidth_range(df, outdir=str(tmp_path))
        assert (tmp_path / "bandwidth_range.png").stat().st_size > 0

    def test_exported_from_init(self, blobs_X, tmp_path):
        from ml_clustering_lab.clustering import estimate_bandwidth_range, plot_bandwidth_range

        df = estimate_bandwidth_range(blobs_X, quantile_range=[0.3, 0.6])
        plot_bandwidth_range(df, outdir=str(tmp_path))
        assert (tmp_path / "bandwidth_range.png").exists()

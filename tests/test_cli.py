"""
tests/test_cli.py
==================
Testes para a interface de linha de comando
(ml_clustering_lab.cli).

Cobertura
---------
- stats: dataset embutido, salvar JSON
- cluster: algoritmo kmeans, dbscan; saída de métricas
- compare: comparação com subset de algoritmos
- dataset --list: lista todos os datasets
- plot: histogram, boxplot, correlation
- Tratamento de erros (dataset inválido, sem args)
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ml_clustering_lab.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStatsCli:
    def test_stats_iris(self):
        result = runner.invoke(app, ["stats", "--dataset", "iris"])
        assert result.exit_code == 0
        assert "Resumo" in result.output or "iris" in result.output

    def test_stats_blobs(self):
        result = runner.invoke(app, ["stats", "--dataset", "blobs"])
        assert result.exit_code == 0

    def test_stats_saves_json(self, tmp_path):
        out = str(tmp_path / "stats.json")
        result = runner.invoke(app, ["stats", "--dataset", "iris", "--output", out])
        assert result.exit_code == 0
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "dataset" in data
        assert "shape" in data

    def test_stats_no_args_exits_nonzero(self):
        result = runner.invoke(app, ["stats"])
        assert result.exit_code != 0 or "Especifique" in result.output

    def test_stats_invalid_dataset(self):
        result = runner.invoke(app, ["stats", "--dataset", "nonexistent_ds_xyz"])
        assert result.exit_code != 0 or "não reconhecido" in result.output or "Erro" in result.output


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


class TestClusterCli:
    def test_kmeans_blobs(self, tmp_path):
        result = runner.invoke(
            app,
            ["cluster", "--dataset", "blobs", "--algorithm", "kmeans", "--n-clusters", "3",
             "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Clusters" in result.output or "Silhouette" in result.output

    def test_dbscan_blobs(self, tmp_path):
        result = runner.invoke(
            app,
            ["cluster", "--dataset", "blobs", "--algorithm", "dbscan",
             "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_cluster_shows_external_metrics_for_iris(self, tmp_path):
        result = runner.invoke(
            app,
            ["cluster", "--dataset", "iris", "--algorithm", "kmeans", "--n-clusters", "3",
             "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        # iris has ground truth, so external metrics should appear
        assert "Métricas Externas" in result.output or "Adjusted Rand" in result.output

    def test_cluster_no_args_exits_nonzero(self):
        result = runner.invoke(app, ["cluster"])
        assert result.exit_code != 0 or "Especifique" in result.output

    def test_cluster_new_scenario_anisotropic(self, tmp_path):
        result = runner.invoke(
            app,
            ["cluster", "--dataset", "anisotropic", "--algorithm", "kmeans",
             "--n-clusters", "3", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_cluster_dbscan_with_eps(self, tmp_path):
        result = runner.invoke(
            app,
            ["cluster", "--dataset", "blobs", "--algorithm", "dbscan",
             "--eps", "0.5", "--min-samples", "5", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareCli:
    def test_compare_iris_kmeans(self, tmp_path):
        result = runner.invoke(
            app,
            ["compare", "--dataset", "iris", "--algorithms", "kmeans",
             "--n-clusters", "3", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_compare_two_algorithms(self, tmp_path):
        result = runner.invoke(
            app,
            ["compare", "--dataset", "blobs", "--algorithms", "kmeans,dbscan",
             "--n-clusters", "3", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_compare_no_args_exits_nonzero(self):
        result = runner.invoke(app, ["compare"])
        assert result.exit_code != 0 or "Especifique" in result.output


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class TestDatasetCli:
    def test_list_shows_iris(self):
        result = runner.invoke(app, ["dataset", "--list"])
        assert result.exit_code == 0
        assert "iris" in result.output

    def test_list_shows_synthetic_scenarios(self):
        result = runner.invoke(app, ["dataset", "--list"])
        assert result.exit_code == 0
        for name in ["blobs", "moons", "circles", "anisotropic", "varied_density", "no_structure"]:
            assert name in result.output

    def test_no_args_shows_hint(self):
        result = runner.invoke(app, ["dataset"])
        assert result.exit_code == 0
        assert "--list" in result.output or "list" in result.output.lower()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


class TestPlotCli:
    def test_plot_histogram(self, tmp_path):
        result = runner.invoke(
            app,
            ["plot", "--dataset", "iris", "--type", "histogram", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "histogram.png").exists()

    def test_plot_boxplot(self, tmp_path):
        result = runner.invoke(
            app,
            ["plot", "--dataset", "iris", "--type", "boxplot", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "boxplot.png").exists()

    def test_plot_correlation(self, tmp_path):
        result = runner.invoke(
            app,
            ["plot", "--dataset", "iris", "--type", "correlation", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert (tmp_path / "correlation.png").exists()

    def test_plot_all(self, tmp_path):
        result = runner.invoke(
            app,
            ["plot", "--dataset", "iris", "--type", "all", "--outdir", str(tmp_path)],
        )
        assert result.exit_code == 0
        for fname in ["histogram.png", "boxplot.png", "correlation.png"]:
            assert (tmp_path / fname).exists()

    def test_plot_invalid_type(self, tmp_path):
        result = runner.invoke(
            app,
            ["plot", "--dataset", "iris", "--type", "pairplot", "--outdir", str(tmp_path)],
        )
        assert result.exit_code != 0 or "não suportado" in result.output

    def test_plot_no_args_exits_nonzero(self):
        result = runner.invoke(app, ["plot"])
        assert result.exit_code != 0 or "Especifique" in result.output

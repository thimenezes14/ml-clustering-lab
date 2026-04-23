"""
tests/test_pipeline.py
=======================
Testes de integração para os pipelines
(ml_clustering_lab.pipeline.run_single e run_compare).

Cobertura
---------
- run_single: execução com dataset embutido (blobs), retorno esperado
- run_single: métricas externas quando há coluna target (iris)
- run_single: novos cenários sintéticos (anisotropic, varied_density, no_structure)
- run_single: ValueError para dataset inválido
- run_compare: execução com subset de algoritmos
- run_compare: retorna DataFrame com colunas esperadas
- run_compare: novos cenários sintéticos
"""

import numpy as np
import pandas as pd
import pytest


class TestRunSingle:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        for key in ["algorithm", "dataset", "labels", "metrics", "elapsed_time", "artifacts"]:
            assert key in result

    def test_labels_shape(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        assert isinstance(result["labels"], np.ndarray)

    def test_metrics_has_silhouette(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        assert "silhouette" in result["metrics"]

    def test_elapsed_time_positive(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        assert result["elapsed_time"] > 0

    def test_artifacts_files_exist(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        import os
        # metrics.json must exist
        assert os.path.exists(result["artifacts"]["metrics"])

    def test_external_metrics_for_sklearn_dataset(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="iris", n_clusters=3, outdir=str(tmp_path))
        # iris has a 'target' column, so external metrics should be computed
        assert "external_metrics" in result
        ext = result["external_metrics"]
        assert "adjusted_rand_index" in ext
        assert "v_measure" in ext

    def test_no_external_metrics_for_synthetic(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        # blobs from load_synthetic has 'label' column, not 'target'
        # so external_metrics should be empty dict
        assert result.get("external_metrics") == {}

    def test_invalid_dataset_raises(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        with pytest.raises(ValueError, match="não reconhecido"):
            run_single("kmeans", dataset="nonexistent_ds", outdir=str(tmp_path))

    def test_no_dataset_raises(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        with pytest.raises(ValueError):
            run_single("kmeans", outdir=str(tmp_path))

    @pytest.mark.parametrize("kind", ["anisotropic", "varied_density"])
    def test_new_synthetic_scenarios(self, kind, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("kmeans", dataset=kind, n_clusters=3, outdir=str(tmp_path / kind))
        assert isinstance(result["labels"], np.ndarray)

    def test_no_structure_scenario(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single(
            "kmeans", dataset="no_structure", n_clusters=3, outdir=str(tmp_path)
        )
        assert isinstance(result["labels"], np.ndarray)

    def test_dbscan_algorithm(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        result = run_single("dbscan", dataset="blobs", outdir=str(tmp_path))
        assert result["algorithm"] == "dbscan"

    def test_run_via_source(self, tmp_path):
        from ml_clustering_lab.pipeline.run_single import run_single

        # Create a minimal CSV file
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n1.5,2.5\n3.5,4.5\n")
        result = run_single("kmeans", source=str(csv_path), n_clusters=2, outdir=str(tmp_path / "out"))
        assert isinstance(result["labels"], np.ndarray)


class TestRunCompare:
    def test_returns_dataframe(self, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        df = run_compare(algorithms=["kmeans"], dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        df = run_compare(algorithms=["kmeans"], dataset="blobs", n_clusters=3, outdir=str(tmp_path))
        for col in ["algorithm", "silhouette", "davies_bouldin", "elapsed_time"]:
            assert col in df.columns

    def test_multiple_algorithms(self, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        df = run_compare(
            algorithms=["kmeans", "dbscan"],
            dataset="blobs",
            n_clusters=3,
            outdir=str(tmp_path),
        )
        assert len(df) == 2

    def test_sorted_by_silhouette(self, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        df = run_compare(
            algorithms=["kmeans", "dbscan"],
            dataset="blobs",
            n_clusters=3,
            outdir=str(tmp_path),
        )
        sil = df["silhouette"].dropna().values
        if len(sil) >= 2:
            assert sil[0] >= sil[-1]

    def test_no_dataset_raises(self, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        with pytest.raises(ValueError):
            run_compare(algorithms=["kmeans"], outdir=str(tmp_path))

    @pytest.mark.parametrize("kind", ["anisotropic", "varied_density"])
    def test_new_synthetic_scenarios(self, kind, tmp_path):
        from ml_clustering_lab.pipeline.run_compare import run_compare

        df = run_compare(
            algorithms=["kmeans"],
            dataset=kind,
            n_clusters=3,
            outdir=str(tmp_path / kind),
        )
        assert isinstance(df, pd.DataFrame)

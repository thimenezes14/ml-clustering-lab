"""
tests/test_loaders.py
=====================
Testes para o módulo de carregamento de datasets (ml_clustering_lab.datasets.loaders).

Cobertura
---------
- Carregamento de CSV local (arquivo existente e inexistente)
- Carregamento de dataset sklearn (iris, wine e nome inválido)
- Geração de dataset sintético (blobs, moons, circles e tipo inválido)
- DatasetRegistry (register, get, list_names)
"""

import pandas as pd
import pytest

from ml_clustering_lab.datasets.loaders import (
    load_csv,
    load_sklearn,
    load_synthetic,
)
from ml_clustering_lab.datasets.registry import DatasetInfo, DatasetRegistry


class TestLoadCsv:
    """Testes para load_csv."""

    def test_loads_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        df = load_csv(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["a", "b"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/tmp/nao_existe_realmente.csv")

    def test_returns_dataframe(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x,y\n0.1,0.2\n0.3,0.4\n")
        df = load_csv(str(f))
        assert isinstance(df, pd.DataFrame)


class TestLoadSklearn:
    """Testes para load_sklearn."""

    def test_iris_shape(self):
        df = load_sklearn("iris")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150

    def test_iris_has_target(self):
        df = load_sklearn("iris")
        assert "target" in df.columns

    def test_wine_shape(self):
        df = load_sklearn("wine")
        assert df.shape[0] == 178

    def test_invalid_dataset(self):
        with pytest.raises(ValueError, match="não suportado"):
            load_sklearn("dataset_que_nao_existe")

    @pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer", "digits"])
    def test_all_supported_datasets(self, name):
        df = load_sklearn(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLoadSynthetic:
    """Testes para load_synthetic."""

    @pytest.mark.parametrize("kind", ["blobs", "moons", "circles"])
    def test_returns_dataframe(self, kind):
        df = load_synthetic(kind=kind, n_samples=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_blobs_has_label_column(self):
        df = load_synthetic(kind="blobs", n_samples=50)
        assert "label" in df.columns

    def test_moons_two_columns(self):
        df = load_synthetic(kind="moons", n_samples=50)
        # x0, x1, label
        assert df.shape[1] == 3

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="não suportado"):
            load_synthetic(kind="hexagons")


class TestDatasetRegistry:
    """Testes para DatasetRegistry."""

    def test_list_names_includes_defaults(self):
        reg = DatasetRegistry()
        names = reg.list_names()
        assert "iris" in names
        assert "blobs" in names
        assert "moons" in names

    def test_get_iris(self):
        reg = DatasetRegistry()
        info = reg.get("iris")
        assert info.name == "iris"
        assert info.source == "sklearn"

    def test_get_unknown_raises_key_error(self):
        reg = DatasetRegistry()
        with pytest.raises(KeyError):
            reg.get("dataset_inexistente")

    def test_register_new(self):
        reg = DatasetRegistry()
        info = DatasetInfo(name="custom", source="local", description="Teste")
        reg.register(info)
        assert "custom" in reg.list_names()

    def test_register_duplicate_raises(self):
        reg = DatasetRegistry()
        with pytest.raises(ValueError):
            reg.register(DatasetInfo(name="iris", source="sklearn", description="dup"))

    def test_list_names_sorted(self):
        reg = DatasetRegistry()
        names = reg.list_names()
        assert names == sorted(names)

"""
tests/test_loaders.py
=====================
Testes para os módulos de carregamento e geração de datasets
(ml_clustering_lab.datasets.loaders e ml_clustering_lab.datasets.generators).

Cobertura
---------
- Carregamento de CSV local (arquivo existente e inexistente)
- Carregamento de dataset sklearn (iris, wine e nome inválido)
- Geração de dataset sintético via loaders (blobs, moons, circles e tipo inválido)
- DatasetRegistry (register, get, list_names)
- generators.generate: todos os cenários, erro para nome inválido
- generators: funções individuais (blobs, moons, circles, anisotropic,
  varied_density, no_structure)
"""

import numpy as np
import pandas as pd
import pytest

from ml_clustering_lab.datasets.generators import (
    AVAILABLE_SCENARIOS,
    generate,
    generate_anisotropic,
    generate_blobs,
    generate_circles,
    generate_moons,
    generate_no_structure,
    generate_varied_density,
)
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

    def test_new_synthetic_scenarios_registered(self):
        reg = DatasetRegistry()
        for name in ("anisotropic", "varied_density", "no_structure"):
            assert name in reg.list_names()


# ---------------------------------------------------------------------------
# generators.py
# ---------------------------------------------------------------------------


class TestGenerate:
    """Testes para o dispatcher generate()."""

    @pytest.mark.parametrize("kind", AVAILABLE_SCENARIOS)
    def test_all_scenarios_return_dataframe(self, kind):
        df = generate(kind, n_samples=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_invalid_kind_raises_value_error(self):
        with pytest.raises(ValueError, match="não encontrado"):
            generate("hexagons")

    def test_reproducibility(self):
        df1 = generate("blobs", n_samples=50, random_state=0)
        df2 = generate("blobs", n_samples=50, random_state=0)
        df1_vals = df1.drop(columns=["label"], errors="ignore").values
        df2_vals = df2.drop(columns=["label"], errors="ignore").values
        assert (df1_vals == df2_vals).all()

    def test_different_seeds_differ(self):
        df1 = generate("blobs", n_samples=50, random_state=0)
        df2 = generate("blobs", n_samples=50, random_state=99)
        assert not (df1["x0"].values == df2["x0"].values).all()


class TestGenerateBlobs:
    def test_shape(self):
        df = generate_blobs(n_samples=150, n_clusters=3)
        assert df.shape == (150, 3)  # x0, x1, label

    def test_n_clusters(self):
        df = generate_blobs(n_samples=200, n_clusters=5)
        assert df["label"].nunique() == 5

    def test_has_label_column(self):
        df = generate_blobs()
        assert "label" in df.columns

    def test_n_features_more_than_2(self):
        df = generate_blobs(n_samples=100, n_features=4)
        assert df.shape[1] == 5  # x0, x1, x2, x3, label

    def test_cluster_std(self):
        df_tight = generate_blobs(n_samples=200, cluster_std=0.1, random_state=1)
        df_loose = generate_blobs(n_samples=200, cluster_std=5.0, random_state=1)
        # Loose blobs should have higher variance
        assert df_loose["x0"].std() > df_tight["x0"].std()


class TestGenerateMoons:
    def test_shape(self):
        df = generate_moons(n_samples=100)
        assert df.shape == (100, 3)  # x0, x1, label

    def test_two_labels(self):
        df = generate_moons(n_samples=200)
        assert set(df["label"].unique()) == {0, 1}

    def test_noise_zero_is_clean(self):
        df = generate_moons(n_samples=100, noise=0.0, random_state=42)
        assert "label" in df.columns


class TestGenerateCircles:
    def test_shape(self):
        df = generate_circles(n_samples=100)
        assert df.shape == (100, 3)

    def test_two_labels(self):
        df = generate_circles(n_samples=200)
        assert set(df["label"].unique()) == {0, 1}

    def test_factor_range(self):
        # factor must be in (0, 1); check we don't crash with valid values
        df = generate_circles(n_samples=100, factor=0.3)
        assert len(df) == 100


class TestGenerateAnisotropic:
    def test_shape(self):
        df = generate_anisotropic(n_samples=150)
        assert df.shape == (150, 3)

    def test_has_label_column(self):
        df = generate_anisotropic()
        assert "label" in df.columns

    def test_n_clusters(self):
        df = generate_anisotropic(n_samples=300, n_clusters=4)
        assert df["label"].nunique() == 4


class TestGenerateVariedDensity:
    def test_shape(self):
        df = generate_varied_density(n_samples=300)
        assert df.shape == (300, 3)

    def test_three_labels(self):
        df = generate_varied_density(n_samples=300)
        assert df["label"].nunique() == 3

    def test_cluster_sizes_approximately_equal(self):
        df = generate_varied_density(n_samples=300)
        counts = df["label"].value_counts()
        # Each cluster should have ~100 samples
        assert all(80 <= c <= 120 for c in counts)


class TestGenerateNoStructure:
    def test_shape(self):
        df = generate_no_structure(n_samples=100)
        assert df.shape == (100, 2)

    def test_no_label_column(self):
        df = generate_no_structure()
        assert "label" not in df.columns

    def test_n_features(self):
        df = generate_no_structure(n_samples=50, n_features=4)
        assert df.shape[1] == 4

    def test_values_in_range(self):
        df = generate_no_structure(n_samples=500, low=-5.0, high=5.0)
        assert df["x0"].min() >= -5.0
        assert df["x0"].max() <= 5.0


"""
tests/test_loaders.py
=====================
Testes para o módulo de carregamento de datasets (ml_clustering_lab.datasets.loaders).

Cobertura planejada
-------------------
- Carregamento de CSV local (arquivo existente e inexistente)
- Carregamento de dataset sklearn (iris, wine e nome inválido)
- Geração de dataset sintético (blobs, moons, circles e tipo inválido)
- Carregamento de URL (requer mock para evitar dependência de rede)

Nota
----
Todos os testes estão marcados com ``pytest.mark.xfail`` ou são stubs
enquanto as funções retornam ``NotImplementedError``. Remova os ``pytest.raises``
de ``NotImplementedError`` conforme implementar cada função.
"""

import pytest

from ml_clustering_lab.datasets.loaders import (
    load_csv,
    load_sklearn,
    load_synthetic,
)


class TestLoadCsv:
    """Testes para load_csv."""

    def test_raises_not_implemented(self, tmp_path):
        """Verifica que NotImplementedError é levado até a implementação."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        with pytest.raises(NotImplementedError):
            load_csv(str(csv_file))


class TestLoadSklearn:
    """Testes para load_sklearn."""

    def test_raises_not_implemented_iris(self):
        """Verifica que NotImplementedError é levado até a implementação."""
        with pytest.raises(NotImplementedError):
            load_sklearn("iris")

    def test_raises_not_implemented_invalid(self):
        """Verifica comportamento para dataset inválido."""
        with pytest.raises(NotImplementedError):
            load_sklearn("dataset_que_nao_existe")


class TestLoadSynthetic:
    """Testes para load_synthetic."""

    @pytest.mark.parametrize("kind", ["blobs", "moons", "circles"])
    def test_raises_not_implemented(self, kind):
        """Verifica que NotImplementedError é levado até a implementação."""
        with pytest.raises(NotImplementedError):
            load_synthetic(kind=kind)

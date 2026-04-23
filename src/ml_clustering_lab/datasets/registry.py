"""
datasets/registry.py
====================
Registro de datasets disponíveis no projeto.

Responsabilidade
----------------
Manter um catálogo centralizado de todos os datasets conhecidos pelo sistema,
com metadados associados (origem, tipo, número de features, número de amostras,
descrição). Funciona como um "dicionário" de datasets que o CLI e os pipelines
podem consultar pelo nome.

Extensão futura
---------------
- Persistência do registry em arquivo JSON/YAML para datasets customizados
- Integração com data catalogs externos (ex.: DVC, Hugging Face Datasets)
- Suporte a datasets versionados
- Auto-descoberta de arquivos CSV em ``data/raw/``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetInfo:
    """Metadados de um dataset registrado.

    Atributos
    ---------
    name : str
        Identificador único do dataset (usado no CLI e nos pipelines).
    source : str
        Origem do dataset: ``"sklearn"`` | ``"synthetic"`` | ``"local"`` | ``"url"``.
    description : str
        Descrição resumida do dataset.
    n_samples : int | None
        Número aproximado de amostras (None se desconhecido antes do carregamento).
    n_features : int | None
        Número de features (None se desconhecido antes do carregamento).
    loader_kwargs : dict[str, Any]
        Argumentos extras repassados para a função de carregamento.
    """

    name: str
    source: str
    description: str
    n_samples: int | None = None
    n_features: int | None = None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)


class DatasetRegistry:
    """Registro centralizado de datasets disponíveis.

    Permite consultar, registrar e listar datasets conhecidos pelo sistema.

    Uso
    ---
    >>> registry = DatasetRegistry()
    >>> registry.get("iris")
    DatasetInfo(name='iris', source='sklearn', ...)
    >>> registry.list_names()
    ['iris', 'wine', 'breast_cancer', 'digits', 'blobs', 'moons', 'circles']

    Extensão futura
    ---------------
    - Método ``load(name)`` que delega para ``loaders.py`` automaticamente
    - Persistência e importação de registros customizados
    - Validação de duplicatas e conflitos de nome
    """

    def __init__(self) -> None:
        """Inicializa o registry com os datasets embutidos padrão."""
        self._registry: dict[str, DatasetInfo] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registra os datasets embutidos (sklearn + sintéticos).

        Ponto de extensão
        -----------------
        Adicione novos datasets embutidos aqui para que fiquem disponíveis
        automaticamente em toda a aplicação.
        """
        defaults = [
            DatasetInfo(
                name="iris",
                source="sklearn",
                description="Dataset clássico de flores Iris com 3 espécies.",
                n_samples=150,
                n_features=4,
            ),
            DatasetInfo(
                name="wine",
                source="sklearn",
                description="Reconhecimento de tipos de vinho por análise química.",
                n_samples=178,
                n_features=13,
            ),
            DatasetInfo(
                name="breast_cancer",
                source="sklearn",
                description="Classificação de tumores (maligno/benigno) por características.",
                n_samples=569,
                n_features=30,
            ),
            DatasetInfo(
                name="digits",
                source="sklearn",
                description="Imagens 8x8 de dígitos manuscritos (0–9).",
                n_samples=1797,
                n_features=64,
            ),
            DatasetInfo(
                name="blobs",
                source="synthetic",
                description="Clusters gaussianos bem separados (ideal para K-Means).",
                loader_kwargs={"kind": "blobs"},
            ),
            DatasetInfo(
                name="moons",
                source="synthetic",
                description="Duas luas entrelaçadas (ideal para DBSCAN e Mean Shift).",
                loader_kwargs={"kind": "moons"},
            ),
            DatasetInfo(
                name="circles",
                source="synthetic",
                description="Círculos concêntricos (ideal para DBSCAN).",
                loader_kwargs={"kind": "circles"},
            ),
            DatasetInfo(
                name="anisotropic",
                source="synthetic",
                description="Clusters elongados via transformação linear (desafia K-Means).",
                loader_kwargs={"kind": "anisotropic"},
            ),
            DatasetInfo(
                name="varied_density",
                source="synthetic",
                description="Clusters com densidades diferentes (desafia DBSCAN).",
                loader_kwargs={"kind": "varied_density"},
            ),
            DatasetInfo(
                name="no_structure",
                source="synthetic",
                description="Dados aleatórios sem estrutura de clusters (caso controle).",
                loader_kwargs={"kind": "no_structure"},
            ),
        ]
        for info in defaults:
            self.register(info)

    def register(self, info: DatasetInfo) -> None:
        """Registra um novo dataset no catálogo.

        Parâmetros
        ----------
        info : DatasetInfo
            Metadados do dataset a registrar.

        Exceções
        --------
        ValueError
            Se já existir um dataset com o mesmo nome.
        """
        if info.name in self._registry:
            raise ValueError(f"Dataset '{info.name}' já está registrado.")
        self._registry[info.name] = info

    def get(self, name: str) -> DatasetInfo:
        """Retorna os metadados de um dataset pelo nome.

        Parâmetros
        ----------
        name : str
            Nome do dataset (case-insensitive).

        Retorna
        -------
        DatasetInfo

        Exceções
        --------
        KeyError
            Se o dataset não estiver registrado.
        """
        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Dataset '{name}' não encontrado. Disponíveis: {available}")
        return self._registry[key]

    def list_names(self) -> list[str]:
        """Retorna a lista de nomes de datasets registrados.

        Retorna
        -------
        list[str]
            Nomes em ordem alfabética.
        """
        return sorted(self._registry.keys())

    def __repr__(self) -> str:  # pragma: no cover
        return f"DatasetRegistry(datasets={list(self._registry.keys())})"

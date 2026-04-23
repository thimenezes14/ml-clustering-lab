"""
datasets/__init__.py
====================
Submódulo de carregamento e registro de datasets.

Exporta as funções de carregamento de alto nível para uso direto::

    from ml_clustering_lab.datasets import load_sklearn, load_csv

Extensão futura
---------------
- Suporte a Parquet e Excel
- Cache de datasets remotos
- Validação de schema com Pydantic
"""

from ml_clustering_lab.datasets.loaders import (
    load_csv,
    load_from_url,
    load_sklearn,
    load_synthetic,
)

__all__ = [
    "load_csv",
    "load_from_url",
    "load_sklearn",
    "load_synthetic",
]

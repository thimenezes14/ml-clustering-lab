"""
datasets/__init__.py
====================
Submódulo de carregamento e registro de datasets.

Exporta as funções de carregamento e geração de alto nível para uso direto::

    from ml_clustering_lab.datasets import load_sklearn, load_csv, generate

Módulos
-------
- ``loaders.py``    : carregamento de CSV, URL, sklearn e sintéticos básicos
- ``generators.py`` : geração de cenários sintéticos para experimentos de clustering
- ``registry.py``   : catálogo centralizado de datasets disponíveis

Extensão futura
---------------
- Suporte a Parquet e Excel
- Cache de datasets remotos
- Validação de schema com Pydantic
"""

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
    load_from_url,
    load_sklearn,
    load_synthetic,
)

__all__ = [
    # loaders
    "load_csv",
    "load_from_url",
    "load_sklearn",
    "load_synthetic",
    # generators
    "generate",
    "generate_blobs",
    "generate_moons",
    "generate_circles",
    "generate_anisotropic",
    "generate_varied_density",
    "generate_no_structure",
    "AVAILABLE_SCENARIOS",
]

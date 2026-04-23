"""
ml_clustering_lab
=================
Laboratório modular de clustering não supervisionado e análise estatística
descritiva em Python.

Módulos públicos
----------------
- datasets   : carregamento de dados de múltiplas fontes
- preprocessing : limpeza, escalonamento e seleção de features
- stats      : análise estatística descritiva
- visualization : gráficos exploratórios e de resultados
- clustering : algoritmos de clustering (K-Means, DBSCAN, Aglomerativo, Mean Shift)
- pipeline   : orquestração de experimentos
- utils      : IO, logging e helpers gerais
- cli        : interface de linha de comando (Typer)

Uso rápido
----------
>>> from ml_clustering_lab.datasets.loaders import load_sklearn
>>> df = load_sklearn("iris")

Ou via CLI::

    ml-lab --help
"""

__version__ = "0.1.0"
__author__ = "thimenezes14"

__all__: list[str] = []

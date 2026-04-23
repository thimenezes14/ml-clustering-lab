"""
clustering/__init__.py
=======================
Submódulo de algoritmos de clustering.

Responsabilidade
----------------
Exportar todos os algoritmos disponíveis e fornecer um registro simples
para resolução pelo nome (usado pelo CLI e pelos pipelines).

Algoritmos disponíveis
----------------------
- ``KMeansRunner``          : K-Means (baseado em centroides)
- ``DBSCANRunner``          : DBSCAN (baseado em densidade)
- ``AgglomerativeRunner``   : Clustering Aglomerativo Hierárquico
- ``MeanShiftRunner``       : Mean Shift (estimativa de kernel de densidade)

Uso
---
>>> from ml_clustering_lab.clustering import get_algorithm
>>> algo = get_algorithm("kmeans", n_clusters=3)
>>> labels = algo.fit_predict(X)

Extensão futura
---------------
- OPTICS, HDBSCAN, GMM, Spectral Clustering
- Seleção automática de hiperparâmetros
"""

from ml_clustering_lab.clustering.agglomerative import AgglomerativeRunner
from ml_clustering_lab.clustering.dbscan import DBSCANRunner
from ml_clustering_lab.clustering.kmeans import KMeansRunner
from ml_clustering_lab.clustering.mean_shift import MeanShiftRunner
from ml_clustering_lab.clustering.optimal_k import (
    elbow_analysis,
    plot_elbow,
    plot_silhouette_analysis,
    silhouette_analysis,
)

# Registro de algoritmos disponíveis: nome (CLI) → classe
ALGORITHM_REGISTRY: dict[str, type] = {
    "kmeans": KMeansRunner,
    "dbscan": DBSCANRunner,
    "agglomerative": AgglomerativeRunner,
    "mean-shift": MeanShiftRunner,
}


def get_algorithm(name: str, **kwargs):
    """Instancia um algoritmo de clustering pelo nome.

    Parâmetros
    ----------
    name : str
        Nome do algoritmo (case-insensitive).
        Valores válidos: ``kmeans``, ``dbscan``, ``agglomerative``, ``mean-shift``.
    **kwargs :
        Hiperparâmetros repassados para o construtor do algoritmo.

    Retorna
    -------
    ClusteringBase
        Instância do algoritmo solicitado.

    Exceções
    --------
    ValueError
        Se ``name`` não corresponder a nenhum algoritmo registrado.

    Exemplo
    -------
    >>> algo = get_algorithm("kmeans", n_clusters=3)
    >>> labels = algo.fit_predict(X)
    """
    key = name.lower().strip()
    if key not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algoritmo '{name}' não encontrado. Disponíveis: {available}")
    return ALGORITHM_REGISTRY[key](**kwargs)


__all__ = [
    "KMeansRunner",
    "DBSCANRunner",
    "AgglomerativeRunner",
    "MeanShiftRunner",
    "ALGORITHM_REGISTRY",
    "get_algorithm",
    "elbow_analysis",
    "silhouette_analysis",
    "plot_elbow",
    "plot_silhouette_analysis",
]

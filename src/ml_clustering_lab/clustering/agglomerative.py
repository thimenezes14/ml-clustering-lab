"""
clustering/agglomerative.py
============================
Implementação do Clustering Aglomerativo Hierárquico para o ml-clustering-lab.

Descrição do algoritmo
----------------------
O Clustering Aglomerativo é um algoritmo **hierárquico bottom-up**. Ele começa
com cada observação em seu próprio cluster e iterativamente une os dois clusters
mais próximos até que todos os pontos pertençam a um único cluster ou ao número
desejado de clusters seja atingido.

O resultado pode ser visualizado como um **dendrograma**, que mostra toda a
hierarquia de fusões — uma das maiores vantagens do método.

Funcionamento (passo a passo)
------------------------------
1. Inicializa com ``n`` clusters (um por observação)
2. Calcula a matriz de distâncias entre todos os pares de clusters
3. Une os dois clusters com menor distância (conforme o critério de ``linkage``)
4. Atualiza a matriz de distâncias
5. Repete os passos 3-4 até atingir ``n_clusters``

Métodos de linkage
------------------
- ``ward``     : minimiza a variância dentro dos clusters — geralmente o melhor
- ``complete`` : distância máxima entre pontos dos clusters (tende a criar clusters compactos)
- ``average``  : distância média entre todos os pares de pontos
- ``single``   : distância mínima entre pontos (sensível a outliers, cria "correntes")

Parâmetros principais
---------------------
- ``n_clusters`` (int): número desejado de clusters
- ``linkage`` (str): critério de linkage — ``"ward"`` | ``"complete"`` | ``"average"`` | ``"single"``
- ``metric`` (str): métrica de distância (apenas para linkage ≠ ward)

Limitações
----------
- Custo computacional O(n² log n) — inviável para datasets muito grandes (n > 10.000)
- O método ``ward`` funciona apenas com distância euclideana
- Não detecta ruído/outliers (todo ponto é atribuído a algum cluster)
- Uma vez feita a fusão, ela não pode ser desfeita (greedy)

Boas práticas
-------------
- Use sempre ``StandardScaler`` antes
- Para datasets maiores, considere BIRCH ou Mini-Batch K-Means
- Use o dendrograma para escolher o número de clusters
- ``ward`` é geralmente a melhor escolha para dados escalados

Extensão futura
---------------
- Método ``plot_dendrogram()`` integrado ao runner
- Suporte a corte automático do dendrograma por distância máxima
- Integração com scipy para dendrogramas mais detalhados
"""

from __future__ import annotations

import numpy as np

from ml_clustering_lab.clustering.base import ClusteringBase
from ml_clustering_lab.config import (
    DEFAULT_AGGLOMERATIVE_LINKAGE,
    DEFAULT_AGGLOMERATIVE_N_CLUSTERS,
)


class AgglomerativeRunner(ClusteringBase):
    """Runner do Clustering Aglomerativo Hierárquico.

    Encapsula o ``sklearn.cluster.AgglomerativeClustering`` com a interface
    padrão do projeto.

    Parâmetros
    ----------
    n_clusters : int, default=3
        Número de clusters.
    linkage : str, default="ward"
        Critério de fusão: ``"ward"`` | ``"complete"`` | ``"average"`` | ``"single"``.
    metric : str, default="euclidean"
        Métrica de distância (ignorada quando ``linkage="ward"``).

    Exemplo
    -------
    >>> runner = AgglomerativeRunner(n_clusters=4, linkage="complete")
    >>> labels = runner.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_AGGLOMERATIVE_N_CLUSTERS,
        linkage: str = DEFAULT_AGGLOMERATIVE_LINKAGE,
        metric: str = "euclidean",
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self._model = None

    @property
    def name(self) -> str:
        """Nome legível do algoritmo."""
        return "Agglomerative"

    @property
    def supports_noise(self) -> bool:
        """Aglomerativo não detecta ruído; todos os pontos são atribuídos a um cluster."""
        return False

    @property
    def requires_k(self) -> bool:
        """Aglomerativo requer que ``n_clusters`` seja informado."""
        return True

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Executa o clustering aglomerativo e retorna os labels.

        Parâmetros
        ----------
        X : np.ndarray de shape (n_samples, n_features)
            Dados de entrada pré-processados (escalonados, sem nulos).

        Retorna
        -------
        np.ndarray de shape (n_samples,)
            Labels de cluster (inteiros de 0 a n_clusters-1).

        Extensão futura
        ---------------
        - Armazenar a estrutura de árvore (``children_``) para dendrograma
        - Suporte a corte por distância de fusão em vez de n_clusters
        """
        raise NotImplementedError(
            "AgglomerativeRunner.fit_predict ainda não foi implementado."
        )

    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do Aglomerativo.

        Retorna
        -------
        dict
            ``{'n_clusters': int, 'linkage': str, 'metric': str}``
        """
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": self.metric,
        }

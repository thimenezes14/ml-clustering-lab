"""
clustering/kmeans.py
====================
Implementação do algoritmo K-Means para o ml-clustering-lab.

Descrição do algoritmo
----------------------
K-Means é um algoritmo de clustering baseado em **centroides**. Ele particiona
``n`` observações em ``k`` clusters minimizando a soma das distâncias quadradas
de cada ponto ao centroide do seu cluster (inércia).

Funcionamento (passo a passo)
------------------------------
1. **Inicialização**: escolhe ``k`` centroides iniciais (aleatório ou ``k-means++``)
2. **Atribuição**: cada ponto é atribuído ao centroide mais próximo (distância euclideana)
3. **Atualização**: os centroides são recalculados como média dos pontos atribuídos
4. **Convergência**: repete os passos 2-3 até que os centroides não se movam mais
   ou até atingir ``max_iter`` iterações

Parâmetros principais
---------------------
- ``n_clusters`` (int): número de clusters K — o parâmetro mais importante
- ``init`` (str): estratégia de inicialização — ``"k-means++"`` (recomendado) ou ``"random"``
- ``n_init`` (int): número de execuções com inicializações diferentes (retorna a melhor)
- ``max_iter`` (int): número máximo de iterações por execução
- ``random_state`` (int): semente para reprodutibilidade

Limitações
----------
- Requer que o número de clusters ``k`` seja informado a priori
- Assume clusters aproximadamente esféricos e de tamanho similar
- Muito sensível à escala dos dados (sempre escalone antes)
- Sensível a outliers (outliers podem distorcer os centroides)
- Pode convergir para ótimos locais (mitigado com ``n_init > 1``)
- Não detecta ruído nem clusters de forma arbitrária

Boas práticas
-------------
- Use sempre ``StandardScaler`` antes de rodar K-Means
- Escolha ``k`` com o método do cotovelo (elbow) ou Silhouette Score
- Use ``init="k-means++"`` para melhor convergência
- Aumente ``n_init`` para resultados mais estáveis

Extensão futura
---------------
- Método ``elbow_plot()`` para auxiliar na escolha de ``k``
- Suporte a Mini-Batch K-Means para datasets grandes
- Métricas automáticas pós-fit (inércia, silhouette)
"""

from __future__ import annotations

import numpy as np

from ml_clustering_lab.clustering.base import ClusteringBase
from ml_clustering_lab.config import (
    DEFAULT_KMEANS_INIT,
    DEFAULT_KMEANS_MAX_ITER,
    DEFAULT_KMEANS_N_CLUSTERS,
    DEFAULT_KMEANS_RANDOM_STATE,
)


class KMeansRunner(ClusteringBase):
    """Runner do algoritmo K-Means.

    Encapsula o ``sklearn.cluster.KMeans`` com a interface padrão do projeto.

    Parâmetros
    ----------
    n_clusters : int, default=3
        Número de clusters.
    init : str, default="k-means++"
        Estratégia de inicialização dos centroides.
    n_init : int, default=10
        Número de execuções com inicializações diferentes.
    max_iter : int, default=300
        Número máximo de iterações por execução.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Exemplo
    -------
    >>> runner = KMeansRunner(n_clusters=3)
    >>> labels = runner.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_KMEANS_N_CLUSTERS,
        init: str = DEFAULT_KMEANS_INIT,
        n_init: int = 10,
        max_iter: int = DEFAULT_KMEANS_MAX_ITER,
        random_state: int = DEFAULT_KMEANS_RANDOM_STATE,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None  # instanciado em fit_predict

    @property
    def name(self) -> str:
        """Nome legível do algoritmo."""
        return "K-Means"

    @property
    def supports_noise(self) -> bool:
        """K-Means não detecta ruído; todos os pontos são atribuídos a um cluster."""
        return False

    @property
    def requires_k(self) -> bool:
        """K-Means requer que ``n_clusters`` seja informado."""
        return True

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Executa K-Means e retorna os labels de cluster.

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
        - Armazenar inércia final e centroides como atributos do runner
        - Adicionar suporte a Mini-Batch K-Means
        """
        raise NotImplementedError("KMeansRunner.fit_predict ainda não foi implementado.")

    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do K-Means.

        Retorna
        -------
        dict
            ``{'n_clusters': int, 'init': str, 'n_init': int,
               'max_iter': int, 'random_state': int}``
        """
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

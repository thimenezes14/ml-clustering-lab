"""
clustering/dbscan.py
====================
Implementação do algoritmo DBSCAN para o ml-clustering-lab.

Descrição do algoritmo
----------------------
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um
algoritmo de clustering baseado em **densidade**. Ele agrupa pontos que estão
densamente concentrados e identifica como **ruído** os pontos isolados em
regiões de baixa densidade.

Funcionamento (passo a passo)
------------------------------
1. Para cada ponto ``p`` não visitado:
   a. Encontra todos os pontos dentro do raio ``eps`` (vizinhança de ``p``)
   b. Se a vizinhança contém ≥ ``min_samples`` pontos → ``p`` é um **core point**
      - Expande o cluster a partir de ``p``, adicionando vizinhos recursivamente
   c. Se a vizinhança contém < ``min_samples`` pontos → ``p`` é **border** ou **noise**
2. Pontos de ruído recebem label **-1**

Conceitos-chave
---------------
- **Core point**: ponto com ≥ ``min_samples`` vizinhos dentro do raio ``eps``
- **Border point**: ponto dentro da vizinhança de um core point mas sem pontos suficientes
- **Noise point**: ponto que não é core nem border — label = -1

Parâmetros principais
---------------------
- ``eps`` (float): raio de vizinhança — a distância máxima entre dois pontos para
  serem considerados vizinhos. Crucial para a qualidade do resultado.
- ``min_samples`` (int): número mínimo de pontos na vizinhança de ``eps`` para
  um ponto ser considerado core point.
- ``metric`` (str): métrica de distância (padrão: ``"euclidean"``)

Limitações
----------
- Muito sensível à escala dos dados (sempre escalone antes)
- ``eps`` é difícil de escolher — use o gráfico de k-distâncias
- Não funciona bem quando os clusters têm densidades muito diferentes
- Custo computacional pode ser alto sem índice espacial em alta dimensão

Boas práticas
-------------
- Sempre normalize os dados antes com ``StandardScaler``
- Use ``k_distance_analysis(X, k=min_samples)`` e ``plot_k_distance()`` para
  estimar visualmente o ``eps`` — o "cotovelo" da curva sugere o valor ideal
- Comece com ``min_samples`` = 2 × n_features como regra empírica
- Verifique se a proporção de ruído (-1) é razoável

Como descobrir os hiperparâmetros
----------------------------------
- ``min_samples``: use a regra empírica ``2 × n_features`` como ponto de partida
- ``eps``: execute ``k_distance_analysis(X, k=min_samples)`` e chame
  ``plot_k_distance()`` — o valor de k-distância no cotovelo da curva é o
  ``eps`` sugerido::

    from ml_clustering_lab.clustering import k_distance_analysis, plot_k_distance
    kd_df = k_distance_analysis(X, k=5)
    plot_k_distance(kd_df, outdir="outputs/figures")

Extensão futura
---------------
- Suporte ao HDBSCAN (versão hierárquica, mais robusta)
- Parâmetro ``algorithm`` para controlar o índice espacial (ball_tree, kd_tree)
- Detecção automática do cotovelo no gráfico de k-distâncias (método Kneedle)
"""

from __future__ import annotations

import numpy as np

from ml_clustering_lab.clustering.base import ClusteringBase
from ml_clustering_lab.config import DEFAULT_DBSCAN_EPS, DEFAULT_DBSCAN_MIN_SAMPLES


class DBSCANRunner(ClusteringBase):
    """Runner do algoritmo DBSCAN.

    Encapsula o ``sklearn.cluster.DBSCAN`` com a interface padrão do projeto.

    Parâmetros
    ----------
    eps : float, default=0.5
        Raio máximo de vizinhança entre dois pontos.
    min_samples : int, default=5
        Número mínimo de pontos na vizinhança para ser core point.
    metric : str, default="euclidean"
        Métrica de distância usada para calcular vizinhanças.

    Exemplo
    -------
    >>> runner = DBSCANRunner(eps=0.3, min_samples=10)
    >>> labels = runner.fit_predict(X)
    >>> noise_count = (labels == -1).sum()
    """

    def __init__(
        self,
        eps: float = DEFAULT_DBSCAN_EPS,
        min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
        metric: str = "euclidean",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._model = None

    @property
    def name(self) -> str:
        """Nome legível do algoritmo."""
        return "DBSCAN"

    @property
    def supports_noise(self) -> bool:
        """DBSCAN detecta ruído; pontos de ruído recebem label -1."""
        return True

    @property
    def requires_k(self) -> bool:
        """DBSCAN determina o número de clusters automaticamente."""
        return False

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Executa DBSCAN e retorna os labels de cluster.

        Parâmetros
        ----------
        X : np.ndarray de shape (n_samples, n_features)
            Dados de entrada pré-processados (escalonados, sem nulos).

        Retorna
        -------
        np.ndarray de shape (n_samples,)
            Labels de cluster. Valor -1 indica ponto de ruído.

        Extensão futura
        ---------------
        - Armazenar ``core_sample_indices_`` e ``components_`` do modelo
        - Calcular e logar proporção de ruído automaticamente
        """
        from sklearn.cluster import DBSCAN

        self._model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        return self._model.fit_predict(X)

    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do DBSCAN.

        Retorna
        -------
        dict
            ``{'eps': float, 'min_samples': int, 'metric': str}``
        """
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
        }

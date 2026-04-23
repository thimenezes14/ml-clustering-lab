"""
clustering/mean_shift.py
=========================
Implementação do algoritmo Mean Shift para o ml-clustering-lab.

Descrição do algoritmo
----------------------
Mean Shift é um algoritmo de clustering baseado em **estimativa de densidade
por kernel (KDE)**. Ele não requer que o número de clusters seja especificado:
o número de clusters é inferido automaticamente a partir da estrutura de
densidade dos dados.

Funcionamento (passo a passo)
------------------------------
1. Para cada ponto de dados, define uma janela deslizante (kernel) de raio ``bandwidth``
2. Calcula o ponto médio (mean) de todos os pontos dentro da janela
3. Desloca o centro da janela para esse ponto médio
4. Repete os passos 2-3 até a convergência (janela não se move mais)
5. Pontos cujas janelas convergem para o mesmo pico de densidade formam um cluster

O ``bandwidth`` controla a resolução: valores menores → mais clusters;
valores maiores → menos clusters.

Parâmetros principais
---------------------
- ``bandwidth`` (float | None): largura da janela do kernel.
  Se None, é estimado automaticamente com ``sklearn.cluster.estimate_bandwidth()``.
- ``bin_seeding`` (bool): se True, usa grade discreta para acelerar o cálculo
  (recomendado para datasets maiores).
- ``max_iter`` (int): máximo de iterações para convergência.

Limitações
----------
- Custo computacional O(n²) — muito lento para datasets grandes (n > 5.000)
- Sensível ao parâmetro ``bandwidth``
- Pode criar muitos clusters pequenos se ``bandwidth`` for muito pequeno
- Não lida bem com clusters de escalas muito diferentes

Boas práticas
-------------
- Sempre normalize os dados antes com ``StandardScaler``
- Use ``estimate_bandwidth()`` para obter um ponto de partida para ``bandwidth``
- Prefira Mean Shift para datasets pequenos ou médios (n < 5.000)
- Use ``bin_seeding=True`` para acelerar em datasets maiores

Extensão futura
---------------
- Suporte a kernels diferentes (gaussian, flat/uniform, epanechnikov)
- Wrapper para visualização da função de densidade estimada
- Integração com ``bandwidth_selection`` automático por validação cruzada
"""

from __future__ import annotations

import numpy as np

from ml_clustering_lab.clustering.base import ClusteringBase
from ml_clustering_lab.config import DEFAULT_MEAN_SHIFT_BANDWIDTH


class MeanShiftRunner(ClusteringBase):
    """Runner do algoritmo Mean Shift.

    Encapsula o ``sklearn.cluster.MeanShift`` com a interface padrão do projeto.

    Parâmetros
    ----------
    bandwidth : float | None, default=None
        Largura da janela do kernel. Se None, estimado automaticamente.
    bin_seeding : bool, default=False
        Se True, usa grade discreta para semear o algoritmo (mais rápido).
    max_iter : int, default=300
        Número máximo de iterações de convergência.

    Exemplo
    -------
    >>> runner = MeanShiftRunner()
    >>> labels = runner.fit_predict(X)
    >>> n_clusters = len(set(labels))
    """

    def __init__(
        self,
        bandwidth: float | None = DEFAULT_MEAN_SHIFT_BANDWIDTH,
        bin_seeding: bool = False,
        max_iter: int = 300,
    ) -> None:
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.max_iter = max_iter
        self._model = None

    @property
    def name(self) -> str:
        """Nome legível do algoritmo."""
        return "Mean Shift"

    @property
    def supports_noise(self) -> bool:
        """Mean Shift não detecta ruído formalmente; todos os pontos são atribuídos."""
        return False

    @property
    def requires_k(self) -> bool:
        """Mean Shift determina o número de clusters automaticamente."""
        return False

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Executa Mean Shift e retorna os labels de cluster.

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
        - Armazenar ``cluster_centers_`` do modelo treinado
        - Estimar e logar ``bandwidth`` quando None
        """
        from sklearn.cluster import MeanShift

        self._model = MeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=self.bin_seeding,
            max_iter=self.max_iter,
        )
        return self._model.fit_predict(X)

    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do Mean Shift.

        Retorna
        -------
        dict
            ``{'bandwidth': float | None, 'bin_seeding': bool, 'max_iter': int}``
        """
        return {
            "bandwidth": self.bandwidth,
            "bin_seeding": self.bin_seeding,
            "max_iter": self.max_iter,
        }

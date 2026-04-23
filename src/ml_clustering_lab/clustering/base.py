"""
clustering/base.py
==================
Classe base abstrata para todos os algoritmos de clustering do projeto.

Responsabilidade
----------------
Definir a interface comum que todos os algoritmos de clustering devem implementar.
Isso garante que o pipeline comparativo possa tratar todos os algoritmos de forma
uniforme, independentemente de suas particularidades.

Interface obrigatória
---------------------
- ``fit_predict(X)``  : treina e retorna os labels de cluster
- ``get_params()``    : retorna dicionário de hiperparâmetros
- ``name``            : propriedade com o nome legível do algoritmo
- ``supports_noise``  : bool — True se o algoritmo pode retornar label -1 (ruído)
- ``requires_k``      : bool — True se o algoritmo precisa de ``n_clusters``

Extensão futura
---------------
- Método ``fit(X)`` separado de ``predict(X)`` para datasets de teste
- Método ``score(X)`` retornando silhouette por padrão
- Serialização/desserialização do modelo treinado
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ClusteringBase(ABC):
    """Classe base abstrata para algoritmos de clustering.

    Todo algoritmo de clustering do projeto deve herdar desta classe e
    implementar os métodos abstratos definidos aqui. Isso garante uma
    interface uniforme para uso no pipeline comparativo.

    Exemplo de implementação
    ------------------------
    >>> class MeuAlgoritmo(ClusteringBase):
    ...     @property
    ...     def name(self) -> str:
    ...         return "Meu Algoritmo"
    ...
    ...     @property
    ...     def supports_noise(self) -> bool:
    ...         return False
    ...
    ...     @property
    ...     def requires_k(self) -> bool:
    ...         return True
    ...
    ...     def fit_predict(self, X: np.ndarray) -> np.ndarray:
    ...         # implementação aqui
    ...         ...
    ...
    ...     def get_params(self) -> dict:
    ...         return {}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome legível do algoritmo (ex.: "K-Means", "DBSCAN")."""

    @property
    @abstractmethod
    def supports_noise(self) -> bool:
        """True se o algoritmo pode retornar label -1 para pontos de ruído.

        Apenas DBSCAN retorna label -1. Algoritmos como K-Means atribuem
        todos os pontos a algum cluster.
        """

    @property
    @abstractmethod
    def requires_k(self) -> bool:
        """True se o algoritmo precisa que ``n_clusters`` seja fornecido.

        K-Means e Aglomerativo requerem ``n_clusters``.
        DBSCAN e Mean Shift determinam o número de clusters automaticamente.
        """

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Treina o algoritmo e retorna os labels de cluster.

        Parâmetros
        ----------
        X : np.ndarray de shape (n_samples, n_features)
            Dados de entrada pré-processados (escalonados, sem nulos).

        Retorna
        -------
        np.ndarray de shape (n_samples,)
            Array de inteiros com os labels de cluster.
            Para DBSCAN, -1 indica ponto de ruído.

        Notas
        -----
        O array ``X`` deve ser escalonado antes de ser passado para este método.
        Use ``preprocessing.scale_features()`` para garantir isso.
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do algoritmo como dicionário.

        Retorna
        -------
        dict
            Dicionário de hiperparâmetros no formato ``{nome: valor}``.

        Exemplo
        -------
        >>> algo.get_params()
        {'n_clusters': 3, 'init': 'k-means++', 'max_iter': 300}
        """

    def __repr__(self) -> str:  # pragma: no cover
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

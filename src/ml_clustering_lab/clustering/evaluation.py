"""
clustering/evaluation.py
=========================
Métricas de avaliação de qualidade de clustering.

Responsabilidade
----------------
Calcular e consolidar métricas de avaliação dos resultados de clustering.
Como clustering é não supervisionado, as métricas são divididas em:

**Métricas internas** (não precisam de rótulos verdadeiros):
- Silhouette Score         : quão bem separados e coesos são os clusters (-1 a 1; maior = melhor)
- Davies-Bouldin Index     : razão entre dispersão interna e separação entre clusters (menor = melhor)
- Calinski-Harabász Score  : razão entre dispersão entre clusters e dentro dos clusters (maior = melhor)

**Métricas externas** (precisam de rótulos verdadeiros — para datasets com ground truth):
- Adjusted Rand Index      : concordância ajustada entre labels previstos e verdadeiros
- Normalized Mutual Info   : informação mútua normalizada
- Homogeneity / Completeness / V-measure

Extensão futura
---------------
- Gráfico de radar comparando múltiplos algoritmos em todas as métricas
- Relatório automático em Markdown/HTML
- Métricas adicionais: WCSS (inércia), Hopkins Statistic, Gap Statistic
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calcula métricas internas de qualidade de clustering.

    Métricas internas não precisam de rótulos verdadeiros, tornando-as
    adequadas para clustering não supervisionado em dados reais.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados usados no clustering (pré-processados).
    labels : np.ndarray de shape (n_samples,)
        Labels de cluster retornados pelo algoritmo.
        Labels -1 (ruído do DBSCAN) são excluídos do cálculo.

    Retorna
    -------
    dict[str, float]
        Dicionário com:
        - ``silhouette``        : float em [-1, 1] (maior = melhor)
        - ``davies_bouldin``    : float ≥ 0 (menor = melhor)
        - ``calinski_harabasz`` : float ≥ 0 (maior = melhor)
        - ``n_clusters``        : int — número de clusters encontrados
        - ``n_noise``           : int — pontos de ruído (label = -1)

    Notas
    -----
    - Requer pelo menos 2 clusters para calcular silhouette e Davies-Bouldin
    - Pontos com label -1 (ruído do DBSCAN) são excluídos do cálculo

    Extensão futura
    ---------------
    - Incluir inércia (WCSS) para K-Means
    - Incluir Hopkins Statistic para testar tendência de clustering
    """
    raise NotImplementedError("compute_internal_metrics ainda não foi implementado.")


def compute_external_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas externas comparando labels previstos com os verdadeiros.

    Útil para datasets com ground truth conhecido (iris, wine, digits, etc.)
    apenas para fins educacionais — em produção real não há rótulos.

    Parâmetros
    ----------
    labels_true : np.ndarray de shape (n_samples,)
        Rótulos verdadeiros (ground truth).
    labels_pred : np.ndarray de shape (n_samples,)
        Labels preditos pelo algoritmo de clustering.

    Retorna
    -------
    dict[str, float]
        Dicionário com:
        - ``adjusted_rand_index``    : float em [-1, 1] (1 = perfeito)
        - ``normalized_mutual_info`` : float em [0, 1] (1 = perfeito)
        - ``homogeneity``            : float em [0, 1]
        - ``completeness``           : float em [0, 1]
        - ``v_measure``              : float em [0, 1]

    Extensão futura
    ---------------
    - Incluir Fowlkes-Mallows Score
    - Incluir matriz de confusão entre clusters e classes verdadeiras
    """
    raise NotImplementedError("compute_external_metrics ainda não foi implementado.")


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Constrói uma tabela comparativa com métricas de múltiplos algoritmos.

    Parâmetros
    ----------
    results : list[dict]
        Lista de dicionários, um por algoritmo, cada um contendo:
        - ``algorithm``    : str — nome do algoritmo
        - ``n_clusters``   : int
        - ``n_noise``      : int
        - ``silhouette``   : float
        - ``davies_bouldin``: float
        - ``calinski_harabasz``: float
        - ``elapsed_time`` : float — tempo de execução em segundos

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma linha por algoritmo e colunas para cada métrica,
        ordenado por Silhouette Score (decrescente).

    Exemplo
    -------
    >>> table = build_comparison_table(results)
    >>> print(table.to_string())

    Extensão futura
    ---------------
    - Adicionar ranking por coluna
    - Exportar como HTML com highlight condicional
    """
    raise NotImplementedError("build_comparison_table ainda não foi implementado.")

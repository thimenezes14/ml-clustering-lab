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
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_noise = int((labels == -1).sum())
    unique_labels = set(labels) - {-1}
    n_clusters = len(unique_labels)

    # Filter out noise for metric computation
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    result: dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }

    if n_clusters >= 2 and len(X_clean) > n_clusters:
        try:
            result["silhouette"] = float(silhouette_score(X_clean, labels_clean))
        except Exception:
            pass
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X_clean, labels_clean))
        except Exception:
            pass
        try:
            result["calinski_harabasz"] = float(
                calinski_harabasz_score(X_clean, labels_clean)
            )
        except Exception:
            pass

    return result


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
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels_true, labels_pred)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(labels_true, labels_pred)
        ),
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "v_measure": float(v_measure_score(labels_true, labels_pred)),
    }


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
    df = pd.DataFrame(results)
    if "silhouette" in df.columns:
        df = df.sort_values("silhouette", ascending=False).reset_index(drop=True)
    return df

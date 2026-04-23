"""
clustering/hyperparams.py
==========================
Utilitários para seleção de hiperparâmetros de algoritmos que **não** usam ``k``.

Responsabilidade
----------------
Fornecer ferramentas analíticas e visuais para auxiliar na escolha dos
hiperparâmetros principais do DBSCAN (``eps``) e do Mean Shift (``bandwidth``),
complementando o módulo ``optimal_k`` que cobre algoritmos baseados em ``k``.

Métodos disponíveis
-------------------
- **Gráfico de k-distâncias (DBSCAN)**: para cada ponto calcula a distância ao
  k-ésimo vizinho mais próximo (k = ``min_samples``). As distâncias ordenadas
  formam uma curva — o "cotovelo" dessa curva sugere o ``eps`` ideal.
- **Faixa de bandwidth (Mean Shift)**: calcula o bandwidth estimado pelo sklearn
  (``estimate_bandwidth``) para diferentes valores de ``quantile``, permitindo
  visualizar como o parâmetro varia e escolher um valor adequado.

Uso
---
>>> from ml_clustering_lab.clustering.hyperparams import (
...     k_distance_analysis,
...     estimate_bandwidth_range,
... )
>>> # DBSCAN: escolha eps
>>> kd_df = k_distance_analysis(X, k=5)
>>> # Mean Shift: escolha bandwidth
>>> bw_df = estimate_bandwidth_range(X)

Extensão futura
---------------
- Detecção automática do cotovelo no gráfico de k-distâncias (segunda derivada)
- Suporte a métricas alternativas para k-distâncias (manhattan, cosine)
- Método de seleção de bandwidth por validação cruzada
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def k_distance_analysis(
    X: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Calcula as k-distâncias de cada ponto para auxiliar na escolha de ``eps`` no DBSCAN.

    Para cada ponto em ``X``, calcula a distância ao seu k-ésimo vizinho mais
    próximo. As distâncias são ordenadas de forma crescente e plotadas —
    o "cotovelo" da curva resultante sugere um bom valor para ``eps``.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados pré-processados (escalonados, sem nulos).
    k : int, default=5
        Número de vizinhos. Deve coincidir com o ``min_samples`` pretendido para
        o DBSCAN. Regra empírica: ``k = 2 × n_features``.
    metric : str, default="euclidean"
        Métrica de distância (mesma que será usada no DBSCAN).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``sample_rank`` (0-indexado, ordenado por
        k-distância crescente) e ``k_distance``.

    Exemplo
    -------
    >>> kd_df = k_distance_analysis(X, k=5)
    >>> # O eps sugerido fica no cotovelo da curva de k_distance
    >>> suggested_eps = kd_df["k_distance"].iloc[int(len(kd_df) * 0.9)]

    Extensão futura
    ---------------
    - Detecção automática do cotovelo via segunda derivada ou método de Kneedle
    - Suporte a índices espaciais (ball_tree, kd_tree) para datasets grandes
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    return pd.DataFrame(
        {
            "sample_rank": np.arange(len(k_distances)),
            "k_distance": k_distances.astype(float),
        }
    )


def plot_k_distance(
    kd_df: pd.DataFrame,
    title: str = "Gráfico de k-Distâncias (DBSCAN — escolha de eps)",
    outdir: str | None = None,
) -> None:
    """Plota a curva de k-distâncias para auxiliar na escolha de ``eps``.

    O "cotovelo" da curva (ponto de maior curvatura) representa a fronteira
    entre pontos que são core points e pontos de ruído. O valor de ``k_distance``
    nesse ponto é um bom candidato para ``eps``.

    Parâmetros
    ----------
    kd_df : pd.DataFrame
        DataFrame retornado por ``k_distance_analysis()``, com colunas
        ``sample_rank`` e ``k_distance``.
    title : str, default="Gráfico de k-Distâncias (DBSCAN — escolha de eps)"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar o gráfico como ``k_distance.png``.
        Se None, exibe na tela.

    Exemplo
    -------
    >>> kd_df = k_distance_analysis(X, k=5)
    >>> plot_k_distance(kd_df, outdir="outputs/figures/meu_dataset")

    Extensão futura
    ---------------
    - Marcar automaticamente o cotovelo com uma linha horizontal
    - Adicionar anotação com o valor de eps sugerido
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml_clustering_lab.utils.io import ensure_dir

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        kd_df["sample_rank"],
        kd_df["k_distance"],
        color="mediumseagreen",
        linewidth=1.5,
    )
    ax.set_xlabel("Pontos ordenados por k-distância")
    ax.set_ylabel("k-distância")
    ax.set_title(title)
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "k_distance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def estimate_bandwidth_range(
    X: np.ndarray,
    quantile_range: list[float] | None = None,
    n_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estima o bandwidth do Mean Shift para diferentes valores de quantile.

    ``sklearn.cluster.estimate_bandwidth(X, quantile=q)`` calcula o bandwidth
    como a distância média ao ``q``-ésimo percentil dos vizinhos. Esta função
    avalia o resultado para uma faixa de valores de ``quantile``, permitindo
    visualizar como o bandwidth varia e escolher um valor adequado.

    Valores menores de ``quantile`` → bandwidth menor → mais clusters.
    Valores maiores de ``quantile`` → bandwidth maior → menos clusters.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados pré-processados (escalonados, sem nulos).
    quantile_range : list[float] | None, default=None
        Valores de quantile a avaliar (entre 0 e 1). Se None, usa
        ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.
    n_samples : int | None, default=None
        Subconjunto de amostras usado para a estimativa. Se None, usa todas.
    random_state : int, default=42
        Semente usada quando ``n_samples`` é fornecido (amostragem aleatória).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``quantile`` e ``bandwidth``, ordenado por
        ``quantile`` crescente.

    Exemplo
    -------
    >>> bw_df = estimate_bandwidth_range(X)
    >>> # Escolha o quantile onde o bandwidth parece estabilizar
    >>> print(bw_df)
       quantile  bandwidth
    0       0.1   0.312...
    1       0.2   0.487...

    Extensão futura
    ---------------
    - Seleção automática do quantile ótimo via validação cruzada
    - Suporte a kernels alternativos (gaussian, epanechnikov)
    """
    from sklearn.cluster import estimate_bandwidth

    if quantile_range is None:
        quantile_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for q in quantile_range:
        bw = estimate_bandwidth(X, quantile=q, n_samples=n_samples, random_state=random_state)
        results.append({"quantile": float(q), "bandwidth": float(bw)})

    return pd.DataFrame(results).sort_values("quantile").reset_index(drop=True)


def plot_bandwidth_range(
    bw_df: pd.DataFrame,
    title: str = "Bandwidth estimado por quantile (Mean Shift)",
    outdir: str | None = None,
) -> None:
    """Plota o bandwidth estimado em função do quantile para auxiliar na escolha.

    A curva mostra como o bandwidth aumenta com o quantile. Escolha um ponto
    onde a curva muda de inclinação (ou use o valor padrão do sklearn para
    uma estimativa automática). Um bandwidth muito pequeno cria muitos clusters
    fragmentados; um muito grande produz poucos clusters grandes.

    Parâmetros
    ----------
    bw_df : pd.DataFrame
        DataFrame retornado por ``estimate_bandwidth_range()``, com colunas
        ``quantile`` e ``bandwidth``.
    title : str, default="Bandwidth estimado por quantile (Mean Shift)"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar o gráfico como ``bandwidth_range.png``.
        Se None, exibe na tela.

    Exemplo
    -------
    >>> bw_df = estimate_bandwidth_range(X)
    >>> plot_bandwidth_range(bw_df, outdir="outputs/figures/meu_dataset")

    Extensão futura
    ---------------
    - Marcar automaticamente o quantile padrão (0.3) com uma linha vertical
    - Adicionar segundo eixo mostrando o número estimado de clusters
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml_clustering_lab.utils.io import ensure_dir

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        bw_df["quantile"],
        bw_df["bandwidth"],
        marker="o",
        color="mediumpurple",
        linewidth=2,
    )
    ax.set_xlabel("quantile")
    ax.set_ylabel("bandwidth estimado")
    ax.set_title(title)
    ax.set_xticks(bw_df["quantile"])
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "bandwidth_range.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

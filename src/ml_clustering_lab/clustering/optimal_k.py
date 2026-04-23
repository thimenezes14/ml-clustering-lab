"""
clustering/optimal_k.py
=======================
Utilitários para determinar o número ótimo de clusters k.

Responsabilidade
----------------
Fornecer ferramentas analíticas e visuais para auxiliar na escolha de ``k``
antes de executar algoritmos que exigem esse parâmetro (K-Means, Aglomerativo).

Métodos disponíveis
-------------------
- **Método do Cotovelo (Elbow)**: calcula a inércia (WCSS) para k em
  uma faixa e plota a curva. O ponto onde a redução da inércia desacelera
  (o "cotovelo") sugere o k ótimo.
- **Análise de Silhouette**: calcula o Silhouette Score médio para cada k.
  O k com maior score indica clusters mais coesos e separados.

Uso
---
>>> from ml_clustering_lab.clustering.optimal_k import elbow_analysis, silhouette_analysis
>>> elbow_df = elbow_analysis(X, k_range=range(2, 10))
>>> sil_df = silhouette_analysis(X, k_range=range(2, 10))

Extensão futura
---------------
- Gap Statistic para determinação automática de k
- Hopkins Statistic para detectar se há estrutura real nos dados
- Suporte a outros algoritmos além de K-Means (ex.: GMM via BIC/AIC)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def elbow_analysis(
    X: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula a inércia (WCSS) do K-Means para diferentes valores de k.

    A curva de inércia decrescente forma um "cotovelo" no valor de k que
    representa a melhor relação entre compacidade dos clusters e complexidade
    do modelo. O ponto de inflexão sugere o k ótimo.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados pré-processados (escalonados, sem nulos).
    k_range : range | list[int] | None, default=None
        Valores de k a avaliar. Se None, usa ``range(2, 11)``.
    random_state : int, default=42
        Semente para reprodutibilidade do K-Means.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``k`` e ``inertia``, ordenado por k crescente.

    Exemplo
    -------
    >>> df = elbow_analysis(X, k_range=range(2, 8))
    >>> df.head()
       k      inertia
    0  2  1234.567890
    1  3   456.789012

    Extensão futura
    ---------------
    - Detecção automática do cotovelo via segunda derivada
    - Suporte a Mini-Batch K-Means para datasets grandes
    """
    from sklearn.cluster import KMeans

    if k_range is None:
        k_range = range(2, 11)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(X)
        results.append({"k": int(k), "inertia": float(km.inertia_)})

    return pd.DataFrame(results).sort_values("k").reset_index(drop=True)


def silhouette_analysis(
    X: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula o Silhouette Score médio do K-Means para diferentes valores de k.

    O Silhouette Score mede quão bem cada amostra está dentro do seu cluster
    comparado aos clusters vizinhos. Valores próximos a 1 indicam clusters
    bem separados. O k com maior score é geralmente o melhor.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados pré-processados (escalonados, sem nulos).
    k_range : range | list[int] | None, default=None
        Valores de k a avaliar. Se None, usa ``range(2, 11)``.
    random_state : int, default=42
        Semente para reprodutibilidade do K-Means.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``k`` e ``silhouette``, ordenado por k crescente.

    Exemplo
    -------
    >>> df = silhouette_analysis(X, k_range=range(2, 8))
    >>> best_k = df.loc[df["silhouette"].idxmax(), "k"]

    Extensão futura
    ---------------
    - Incluir desvio padrão do silhouette (por amostra) como indicador de estabilidade
    - Suporte a outros algoritmos de clustering (DBSCAN, Aglomerativo)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = range(2, 11)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        score = float(silhouette_score(X, labels))
        results.append({"k": int(k), "silhouette": score})

    return pd.DataFrame(results).sort_values("k").reset_index(drop=True)


def plot_elbow(
    elbow_df: pd.DataFrame,
    title: str = "Método do Cotovelo",
    outdir: str | None = None,
) -> None:
    """Plota a curva do método do cotovelo (inércia vs k).

    Parâmetros
    ----------
    elbow_df : pd.DataFrame
        DataFrame retornado por ``elbow_analysis()``, com colunas ``k`` e ``inertia``.
    title : str, default="Método do Cotovelo"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar o gráfico como ``elbow.png``. Se None, exibe na tela.

    Exemplo
    -------
    >>> elbow_df = elbow_analysis(X)
    >>> plot_elbow(elbow_df, outdir="outputs/figures/iris")

    Extensão futura
    ---------------
    - Marcar automaticamente o cotovelo detectado com uma linha vertical
    - Adicionar anotação com o valor de inércia para cada k
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml_clustering_lab.utils.io import ensure_dir

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("k (número de clusters)")
    ax.set_ylabel("Inércia (WCSS)")
    ax.set_title(title)
    ax.set_xticks(elbow_df["k"])
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "elbow.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_silhouette_analysis(
    silhouette_df: pd.DataFrame,
    title: str = "Análise de Silhouette por k",
    outdir: str | None = None,
) -> None:
    """Plota o Silhouette Score médio para cada valor de k.

    Parâmetros
    ----------
    silhouette_df : pd.DataFrame
        DataFrame retornado por ``silhouette_analysis()``,
        com colunas ``k`` e ``silhouette``.
    title : str, default="Análise de Silhouette por k"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar o gráfico como ``silhouette_analysis.png``.
        Se None, exibe na tela.

    Exemplo
    -------
    >>> sil_df = silhouette_analysis(X)
    >>> plot_silhouette_analysis(sil_df, outdir="outputs/figures/iris")

    Extensão futura
    ---------------
    - Marcar automaticamente o k com maior silhouette
    - Incluir banda de confiança (desvio padrão por amostra)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ml_clustering_lab.utils.io import ensure_dir

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        silhouette_df["k"],
        silhouette_df["silhouette"],
        marker="o",
        color="darkorange",
        linewidth=2,
    )
    ax.set_xlabel("k (número de clusters)")
    ax.set_ylabel("Silhouette Score médio")
    ax.set_title(title)
    ax.set_xticks(silhouette_df["k"])
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "silhouette_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

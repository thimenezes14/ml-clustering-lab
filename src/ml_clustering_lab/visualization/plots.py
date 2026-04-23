"""
visualization/plots.py
=======================
Funções de visualização exploratória e de resultados de clustering.

Responsabilidade
----------------
Gerar e salvar gráficos em formato PNG para:

- **Exploração de dados** : histogramas, boxplots, heatmap de correlação, pairplot
- **Resultados de clustering** : scatter colorido por cluster, dendrograma
- **Comparação** : gráfico de barras com métricas de múltiplos algoritmos

Todos os gráficos aceitam um parâmetro ``outdir`` opcional; se fornecido,
o arquivo é salvo em disco. Se None, o gráfico é exibido na tela.

Extensão futura
---------------
- Suporte a Plotly para gráficos interativos
- Parâmetro ``figsize`` configurável por gráfico
- Função ``plot_all(df)`` que gera todos os gráficos de uma vez
- Suporte a temas dark/light via matplotlib styles
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_clustering_lab.utils.io import ensure_dir


def plot_histogram(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    outdir: str | None = None,
) -> None:
    """Gera histogramas com KDE para colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    columns : list[str] | None, default=None
        Colunas a plotar. Se None, plota todas as colunas numéricas.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Parâmetro ``hue`` para colorir por categoria
    - Sobreposição de distribuição normal teórica
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if columns is not None:
        numeric_cols = [c for c in columns if c in numeric_cols]
    if not numeric_cols:
        return

    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_boxplot(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    outdir: str | None = None,
) -> None:
    """Gera boxplots para colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    columns : list[str] | None, default=None
        Colunas a plotar. Se None, plota todas as colunas numéricas.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Parâmetro ``hue`` para agrupar por categoria
    - Marcação visual de outliers
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if columns is not None:
        numeric_cols = [c for c in columns if c in numeric_cols]
    if not numeric_cols:
        return

    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(col)
        ax.set_xlabel(col)

    plt.tight_layout()
    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    outdir: str | None = None,
) -> None:
    """Gera heatmap da matriz de correlação.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados (apenas colunas numéricas são usadas).
    method : str, default="pearson"
        Método de correlação: ``"pearson"`` | ``"spearman"`` | ``"kendall"``.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Máscara do triângulo superior para facilitar leitura
    - Destacar correlações acima de um threshold
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return

    corr = numeric_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Correlação ({method})")
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_scatter_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    title: str = "Clustering Result",
    outdir: str | None = None,
) -> None:
    """Gera scatter plot 2D colorido por cluster.

    Se ``X`` tiver mais de 2 features, usa as duas primeiras colunas.
    Para visualização adequada em alta dimensão, use ``plot_pca_2d()``
    do módulo ``embeddings``.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados usados no clustering.
    labels : np.ndarray de shape (n_samples,)
        Labels de cluster (pontos com label -1 são exibidos em cinza).
    feature_names : list[str] | None, default=None
        Nomes das features para os eixos.
    title : str, default="Clustering Result"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Marcar centroides para K-Means
    - Parâmetro ``dim_reduction`` para reduzir automaticamente com PCA
    """
    if X.shape[1] > 2:
        X_plot = X[:, :2]
    else:
        X_plot = X

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl in unique_labels:
        mask = labels == lbl
        color = "gray" if lbl == -1 else cmap(lbl % 10)
        label_str = "Ruído" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[color], label=label_str, s=20, alpha=0.7)

    xname = feature_names[0] if feature_names and len(feature_names) > 0 else "Feature 0"
    yname = feature_names[1] if feature_names and len(feature_names) > 1 else "Feature 1"
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        safe_title = title.lower().replace(" ", "_")
        fig.savefig(Path(outdir) / f"{safe_title}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dendrogram(
    model,
    title: str = "Hierarchical Clustering Dendrogram",
    outdir: str | None = None,
) -> None:
    """Gera dendrograma para modelo de clustering aglomerativo.

    Parâmetros
    ----------
    model : sklearn.cluster.AgglomerativeClustering
        Modelo aglomerativo treinado (com ``compute_full_tree=True``).
    title : str, default="Hierarchical Clustering Dendrogram"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Parâmetro ``truncate_mode`` para controlar exibição em datasets grandes
    - Marcação automática do corte ótimo
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, ax = plt.subplots(figsize=(10, 5))
    # Build linkage matrix from the AgglomerativeClustering model
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, ax=ax, truncate_mode="level", p=5)
    ax.set_title(title)
    ax.set_xlabel("Número de pontos em nó (ou índice se folha)")
    ax.set_ylabel("Distância")
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "dendrogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_compare_metrics(
    comparison_df: pd.DataFrame,
    outdir: str | None = None,
) -> None:
    """Gera gráfico de barras comparando métricas de múltiplos algoritmos.

    Parâmetros
    ----------
    comparison_df : pd.DataFrame
        DataFrame com uma linha por algoritmo e colunas para cada métrica.
        Produzido por ``evaluation.build_comparison_table()``.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Gráfico de radar (spider chart) para comparação multidimensional
    - Destaque automático do melhor algoritmo por métrica
    """
    metric_cols = [c for c in ["silhouette", "davies_bouldin", "calinski_harabasz"] if c in comparison_df.columns]
    if not metric_cols or "algorithm" not in comparison_df.columns:
        return

    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        ax.bar(comparison_df["algorithm"], comparison_df[metric], color="steelblue")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Algoritmo")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "comparison_metrics_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

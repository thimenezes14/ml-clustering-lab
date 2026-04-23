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

import pandas as pd
import numpy as np


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
    raise NotImplementedError("plot_histogram ainda não foi implementado.")


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
    raise NotImplementedError("plot_boxplot ainda não foi implementado.")


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
    raise NotImplementedError("plot_correlation ainda não foi implementado.")


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
    raise NotImplementedError("plot_scatter_clusters ainda não foi implementado.")


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
    raise NotImplementedError("plot_dendrogram ainda não foi implementado.")


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
    raise NotImplementedError("plot_compare_metrics ainda não foi implementado.")

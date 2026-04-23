"""
visualization/embeddings.py
============================
Redução de dimensionalidade para visualização de clusters em 2D.

Responsabilidade
----------------
Quando os dados têm mais de 2 dimensões, é necessário reduzir a dimensionalidade
para visualizar os clusters em um scatter plot 2D. Este módulo fornece wrappers
para as técnicas mais comuns:

- **PCA** (Principal Component Analysis): rápido, linear, boa preservação de variância
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): não-linear, preserva estrutura local
- **UMAP** (Uniform Manifold Approximation and Projection): mais rápido que t-SNE,
  preserva estrutura local e global (requer instalação de ``umap-learn``)

Quando usar cada técnica
------------------------
- **PCA**: para visualização rápida e interpretação de variância explicada
- **t-SNE**: para exploração de estrutura local dos dados; não é determinístico
- **UMAP**: melhor compromisso entre velocidade e qualidade; recomendado para datasets grandes

Extensão futura
---------------
- Suporte a autoencoder para redução não-linear
- Visualização 3D interativa com Plotly
- Parâmetros de perplexidade (t-SNE) e n_neighbors (UMAP) configuráveis
"""

from __future__ import annotations

import numpy as np


def reduce_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Reduz dimensionalidade com PCA.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados de entrada.
    n_components : int, default=2
        Número de componentes principais.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    np.ndarray de shape (n_samples, n_components)
        Dados no espaço de componentes principais.

    Extensão futura
    ---------------
    - Retornar variância explicada por componente
    - Gráfico de scree plot
    """
    raise NotImplementedError("reduce_pca ainda não foi implementado.")


def reduce_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Reduz dimensionalidade com t-SNE.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados de entrada.
    n_components : int, default=2
        Número de dimensões de saída.
    perplexity : float, default=30.0
        Parâmetro de perplexidade do t-SNE. Valores típicos: 5–50.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    np.ndarray de shape (n_samples, n_components)
        Dados no espaço t-SNE.

    Notas
    -----
    t-SNE é computacionalmente caro para n > 5.000. Use PCA primeiro para
    reduzir para ~50 dimensões antes de aplicar t-SNE.

    Extensão futura
    ---------------
    - Parâmetro ``n_iter`` configurável
    - Pré-redução automática com PCA quando n_features > 50
    """
    raise NotImplementedError("reduce_tsne ainda não foi implementado.")


def plot_pca_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA 2D",
    outdir: str | None = None,
) -> None:
    """Reduz para 2D com PCA e plota scatter colorido por cluster.

    Parâmetros
    ----------
    X : np.ndarray de shape (n_samples, n_features)
        Dados de entrada.
    labels : np.ndarray de shape (n_samples,)
        Labels de cluster.
    title : str, default="PCA 2D"
        Título do gráfico.
    outdir : str | None, default=None
        Diretório para salvar. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Incluir variância explicada nos eixos (ex.: "PC1 (45.2%)")
    - Parâmetro ``method`` para escolher entre PCA, t-SNE, UMAP
    """
    raise NotImplementedError("plot_pca_2d ainda não foi implementado.")

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

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml_clustering_lab.utils.io import ensure_dir


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
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


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
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(X)


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
    X_2d = reduce_pca(X, n_components=2)

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl in unique_labels:
        mask = labels == lbl
        color = "gray" if lbl == -1 else cmap(lbl % 10)
        label_str = "Ruído" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label_str, s=20, alpha=0.7)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "pca_2d.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

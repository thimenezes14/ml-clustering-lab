"""
visualization/__init__.py
==========================
Submódulo de visualização de dados e resultados de clustering.

Responsabilidade
----------------
Fornecer funções de alto nível para geração de gráficos exploratórios
(univariados e multivariados) e de resultados de clustering, salvando
os artefatos em disco ou exibindo na tela.

Extensão futura
---------------
- Gráficos interativos com Plotly
- Relatórios automáticos com todas as figuras
- Suporte a temas customizados
"""

from ml_clustering_lab.visualization.plots import (
    plot_boxplot,
    plot_compare_metrics,
    plot_correlation,
    plot_dendrogram,
    plot_histogram,
    plot_scatter_clusters,
)

__all__ = [
    "plot_histogram",
    "plot_boxplot",
    "plot_correlation",
    "plot_scatter_clusters",
    "plot_dendrogram",
    "plot_compare_metrics",
]

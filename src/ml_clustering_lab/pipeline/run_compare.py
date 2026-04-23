"""
pipeline/run_compare.py
========================
Pipeline de execução comparativa de múltiplos algoritmos de clustering.

Responsabilidade
----------------
Orquestrar o fluxo completo de comparação:

1. Carregar e pré-processar o dataset (uma única vez, igual para todos)
2. Executar cada algoritmo solicitado sobre o mesmo dataset pré-processado
3. Coletar métricas de cada execução (silhouette, DBI, CHI, n_clusters, n_noise, tempo)
4. Gerar tabela comparativa
5. Gerar gráficos lado a lado e gráfico de barras de métricas
6. Salvar todos os artefatos em um diretório de saída estruturado

Saídas geradas
--------------
- ``comparison_metrics.csv``          : tabela comparativa de métricas
- ``comparison_metrics.json``         : idem em JSON
- ``comparison_metrics_plot.png``     : gráfico de barras das métricas
- ``clusters_{algorithm}.png``        : scatter dos clusters por algoritmo
- ``pca_2d_{algorithm}.png``          : visualização PCA 2D por algoritmo

Extensão futura
---------------
- Execução paralela dos algoritmos com ``concurrent.futures``
- Suporte a configuração por YAML
- Relatório HTML automático com todas as figuras e métricas
- Integração com MLflow para rastreamento de experimentos
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_compare(
    algorithms: list[str] | None = None,
    dataset: str | None = None,
    source: str | None = None,
    n_clusters: int | None = None,
    outdir: str | Path | None = None,
) -> pd.DataFrame:
    """Executa e compara múltiplos algoritmos de clustering.

    Parâmetros
    ----------
    algorithms : list[str] | None, default=None
        Lista de algoritmos a executar.
        Se None, executa todos os 4 algoritmos disponíveis.
    dataset : str | None, default=None
        Nome de dataset embutido (iris, wine, blobs, etc.).
    source : str | None, default=None
        Caminho para arquivo CSV local ou URL.
    n_clusters : int | None, default=None
        Número de clusters para algoritmos que exigem (K-Means, Aglomerativo).
    outdir : str | Path | None, default=None
        Diretório para salvar artefatos. Se None, usa ``outputs/runs/``.

    Retorna
    -------
    pd.DataFrame
        Tabela comparativa com uma linha por algoritmo e colunas:
        ``algorithm``, ``n_clusters``, ``n_noise``, ``silhouette``,
        ``davies_bouldin``, ``calinski_harabasz``, ``elapsed_time``.

    Exemplo
    -------
    >>> table = run_compare(dataset="iris", algorithms=["kmeans", "dbscan"])
    >>> print(table[["algorithm", "silhouette", "davies_bouldin"]])

    Extensão futura
    ---------------
    - Suporte a parâmetros por algoritmo via dicionário
    - Execução paralela com ``concurrent.futures.ThreadPoolExecutor``
    - Geração automática de relatório em HTML
    """
    raise NotImplementedError("run_compare ainda não foi implementado.")

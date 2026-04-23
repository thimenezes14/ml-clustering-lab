"""
pipeline/run_single.py
=======================
Pipeline de execução de um único experimento de clustering.

Responsabilidade
----------------
Orquestrar o fluxo completo para um único algoritmo:

1. Carregar dataset (disco, URL ou sklearn)
2. Inspecionar e exibir sumário do dataset
3. Limpar dados (nulos, duplicados)
4. Selecionar features numéricas
5. Escalar features
6. Executar algoritmo de clustering
7. Calcular métricas de avaliação
8. Gerar visualizações (scatter, PCA 2D)
9. Salvar artefatos (métricas JSON, gráficos PNG)

Extensão futura
---------------
- Suporte a configuração via dicionário/YAML
- Retornar objeto ``ExperimentResult`` com todos os artefatos
- Rastreamento de experimento com MLflow
"""

from __future__ import annotations

from pathlib import Path


def run_single(
    algorithm: str,
    dataset: str | None = None,
    source: str | None = None,
    n_clusters: int | None = None,
    outdir: str | Path | None = None,
    **algo_kwargs,
) -> dict:
    """Executa um experimento de clustering com um único algoritmo.

    Parâmetros
    ----------
    algorithm : str
        Nome do algoritmo: ``"kmeans"`` | ``"dbscan"`` | ``"agglomerative"`` | ``"mean-shift"``.
    dataset : str | None, default=None
        Nome de dataset embutido (iris, wine, blobs, etc.).
        Mutuamente exclusivo com ``source``.
    source : str | None, default=None
        Caminho para arquivo CSV local ou URL.
        Mutuamente exclusivo com ``dataset``.
    n_clusters : int | None, default=None
        Número de clusters (obrigatório para K-Means e Aglomerativo).
    outdir : str | Path | None, default=None
        Diretório para salvar artefatos. Se None, usa ``outputs/runs/``.
    **algo_kwargs :
        Parâmetros adicionais repassados para o construtor do algoritmo
        (ex.: ``eps=0.3``, ``min_samples=10`` para DBSCAN).

    Retorna
    -------
    dict
        Dicionário com:
        - ``algorithm``     : str
        - ``dataset``       : str
        - ``labels``        : np.ndarray
        - ``metrics``       : dict com métricas internas
        - ``elapsed_time``  : float (segundos)
        - ``artifacts``     : dict com caminhos dos arquivos salvos

    Exceções
    --------
    ValueError
        Se nenhum dataset for especificado ou se ``n_clusters`` não for
        fornecido para algoritmos que o exigem.

    Exemplo
    -------
    >>> result = run_single("kmeans", dataset="iris", n_clusters=3)
    >>> result["metrics"]["silhouette"]
    0.55

    Extensão futura
    ---------------
    - Parâmetro ``config_file`` para carregar configurações de YAML
    - Retorno de objeto tipado ``ExperimentResult``
    """
    raise NotImplementedError("run_single ainda não foi implementado.")

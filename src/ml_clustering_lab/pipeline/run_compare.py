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
    import time

    from ml_clustering_lab.clustering import ALGORITHM_REGISTRY, get_algorithm
    from ml_clustering_lab.clustering.evaluation import build_comparison_table, compute_internal_metrics
    from ml_clustering_lab.config import RUNS_DIR
    from ml_clustering_lab.datasets.loaders import load_csv, load_sklearn, load_synthetic
    from ml_clustering_lab.preprocessing.cleaning import drop_duplicates, drop_missing
    from ml_clustering_lab.preprocessing.feature_selection import select_numeric_features
    from ml_clustering_lab.preprocessing.scaling import scale_features
    from ml_clustering_lab.utils.io import ensure_dir, save_dataframe, save_json
    from ml_clustering_lab.visualization.plots import plot_compare_metrics, plot_scatter_clusters

    if dataset is None and source is None:
        raise ValueError("Especifique 'dataset' ou 'source'.")

    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())

    # --- Load data ---
    if dataset is not None:
        from ml_clustering_lab.datasets import AVAILABLE_SCENARIOS

        _SKLEARN_NAMES = {"iris", "wine", "breast_cancer", "digits"}
        _SYNTHETIC_NAMES = set(AVAILABLE_SCENARIOS)
        if dataset.lower() in _SKLEARN_NAMES:
            df = load_sklearn(dataset)
        elif dataset.lower() in _SYNTHETIC_NAMES:
            df = load_synthetic(kind=dataset)
        else:
            raise ValueError(f"Dataset embutido '{dataset}' não reconhecido.")
        dataset_name = dataset
    else:
        assert source is not None
        if source.startswith("http://") or source.startswith("https://"):
            from ml_clustering_lab.datasets.loaders import load_from_url
            df = load_from_url(source)
        else:
            df = load_csv(source)
        dataset_name = Path(source).stem

    # --- Preprocess once ---
    df = drop_missing(df)
    df = drop_duplicates(df)
    exclude_cols = [c for c in ["target", "label"] if c in df.columns]
    X_df = select_numeric_features(df, exclude=exclude_cols)
    X_df = scale_features(X_df)
    X = X_df.values

    # --- Output directory ---
    out_path = Path(outdir) if outdir else RUNS_DIR / f"compare_{dataset_name}"
    ensure_dir(out_path)

    # --- Run each algorithm ---
    results = []
    for algo_name in algorithms:
        kwargs: dict = {}
        if n_clusters is not None:
            kwargs["n_clusters"] = n_clusters
        try:
            algo = get_algorithm(algo_name, **kwargs)
            t0 = time.time()
            labels = algo.fit_predict(X)
            elapsed = time.time() - t0
            metrics = compute_internal_metrics(X, labels)
            metrics["algorithm"] = algo.name
            metrics["elapsed_time"] = round(elapsed, 4)
            results.append(metrics)
            # Save scatter
            plot_scatter_clusters(
                X, labels,
                title=f"{algo.name} — {dataset_name}",
                outdir=str(out_path),
            )
        except Exception as exc:
            results.append({
                "algorithm": algo_name,
                "n_clusters": float("nan"),
                "n_noise": float("nan"),
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
                "elapsed_time": float("nan"),
                "error": str(exc),
            })

    comparison_df = build_comparison_table(results)
    save_dataframe(comparison_df, out_path / "comparison_metrics.csv")
    save_json(results, out_path / "comparison_metrics.json")
    plot_compare_metrics(comparison_df, outdir=str(out_path))

    return comparison_df

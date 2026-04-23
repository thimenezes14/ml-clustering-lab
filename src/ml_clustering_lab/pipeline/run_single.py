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
    import time

    from ml_clustering_lab.clustering import get_algorithm
    from ml_clustering_lab.clustering.evaluation import compute_internal_metrics
    from ml_clustering_lab.config import RUNS_DIR
    from ml_clustering_lab.datasets.loaders import load_csv, load_sklearn, load_synthetic
    from ml_clustering_lab.preprocessing.cleaning import drop_duplicates, drop_missing
    from ml_clustering_lab.preprocessing.feature_selection import select_numeric_features
    from ml_clustering_lab.preprocessing.scaling import scale_features
    from ml_clustering_lab.utils.io import ensure_dir, save_json
    from ml_clustering_lab.visualization.embeddings import plot_pca_2d
    from ml_clustering_lab.visualization.plots import plot_scatter_clusters

    if dataset is None and source is None:
        raise ValueError("Especifique 'dataset' ou 'source'.")

    # --- Load data ---
    if dataset is not None:
        # Try sklearn datasets first, then synthetic
        _SKLEARN_NAMES = {"iris", "wine", "breast_cancer", "digits"}
        _SYNTHETIC_NAMES = {"blobs", "moons", "circles"}
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

    # --- Preprocess ---
    df = drop_missing(df)
    df = drop_duplicates(df)

    # Exclude 'target' and 'label' from features
    exclude_cols = [c for c in ["target", "label"] if c in df.columns]
    X_df = select_numeric_features(df, exclude=exclude_cols)
    X_df = scale_features(X_df)
    X = X_df.values

    # --- Build algorithm kwargs ---
    kwargs = dict(algo_kwargs)
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters

    algo = get_algorithm(algorithm, **kwargs)

    # --- Run clustering ---
    t0 = time.time()
    labels = algo.fit_predict(X)
    elapsed = time.time() - t0

    # --- Evaluate ---
    metrics = compute_internal_metrics(X, labels)
    metrics["elapsed_time"] = round(elapsed, 4)

    # --- Save artifacts ---
    out_path = Path(outdir) if outdir else RUNS_DIR / f"{algorithm}_{dataset_name}"
    ensure_dir(out_path)

    save_json({**metrics, "algorithm": algorithm, "dataset": dataset_name}, out_path / "metrics.json")
    plot_scatter_clusters(X, labels, title=f"{algo.name} — {dataset_name}", outdir=str(out_path))
    plot_pca_2d(X, labels, title=f"PCA 2D — {algo.name} — {dataset_name}", outdir=str(out_path))

    return {
        "algorithm": algorithm,
        "dataset": dataset_name,
        "labels": labels,
        "metrics": metrics,
        "elapsed_time": elapsed,
        "artifacts": {
            "metrics": str(out_path / "metrics.json"),
            "scatter": str(out_path / f"{algo.name.lower().replace(' ', '_')}___{dataset_name}.png"),
            "pca_2d": str(out_path / "pca_2d.png"),
        },
    }

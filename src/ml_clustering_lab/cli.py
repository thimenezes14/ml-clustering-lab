"""
cli.py
======
Interface de linha de comando para o ml-clustering-lab.

Esta CLI é construída com `Typer` e oferece cinco comandos principais:

- ``stats``   — análise estatística descritiva de um dataset
- ``cluster`` — executa um algoritmo de clustering
- ``compare`` — compara múltiplos algoritmos no mesmo dataset
- ``dataset`` — lista ou carrega datasets disponíveis
- ``plot``    — gera visualizações para um dataset

Uso
---
Após instalar o pacote::

    ml-lab --help
    ml-lab stats --dataset iris
    ml-lab cluster --dataset iris --algorithm kmeans --n-clusters 3
    ml-lab compare --dataset iris
    ml-lab dataset --list
    ml-lab plot --dataset iris

Extensão futura
---------------
- Suporte a flags globais como ``--verbose`` e ``--output-dir``
- Subcomandos para experiment tracking
- Geração automática de relatório em HTML/Markdown
- Integração com arquivos de configuração YAML
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="ml-lab",
    help=(
        "ml-clustering-lab — laboratório de clustering não supervisionado "
        "e análise estatística descritiva."
    ),
    add_completion=False,
)

console = Console()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_dataframe(dataset: Optional[str], source: Optional[str]):
    """Carrega um DataFrame a partir de dataset embutido ou arquivo CSV."""
    from ml_clustering_lab.datasets import AVAILABLE_SCENARIOS
    from ml_clustering_lab.datasets.loaders import load_csv, load_from_url, load_sklearn, load_synthetic

    _SKLEARN_NAMES = {"iris", "wine", "breast_cancer", "digits"}
    _SYNTHETIC_NAMES = set(AVAILABLE_SCENARIOS)

    if dataset is not None:
        if dataset.lower() in _SKLEARN_NAMES:
            return load_sklearn(dataset), dataset
        elif dataset.lower() in _SYNTHETIC_NAMES:
            return load_synthetic(kind=dataset), dataset
        else:
            console.print(f"[red]Dataset '{dataset}' não reconhecido.[/red]")
            raise typer.Exit(code=1)
    elif source is not None:
        if source.startswith("http://") or source.startswith("https://"):
            df = load_from_url(source)
        else:
            df = load_csv(source)
        import re
        name = re.sub(r"[^a-zA-Z0-9_]", "_", source.rstrip("/").split("/")[-1].rsplit(".", 1)[0])
        return df, name
    else:
        console.print("[red]Especifique --dataset ou --source.[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Nome de um dataset embutido (ex.: iris, wine, blobs).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Caminho para um arquivo CSV local.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Caminho para salvar o relatório JSON de estatísticas.",
    ),
) -> None:
    """Exibe análise estatística descritiva de um dataset.

    Calcula e exibe medidas de tendência central (média, mediana, moda),
    dispersão (variância, desvio padrão, IQR), forma (skewness, kurtosis),
    qualidade dos dados (nulos, duplicados, outliers) e distribuições.

    Exemplos::

        ml-lab stats --dataset iris
        ml-lab stats --source data/raw/meus_dados.csv --output outputs/reports/stats.json
    """
    from ml_clustering_lab.stats.descriptive import (
        central_tendency,
        describe_dataframe,
        detect_outliers,
        dispersion,
        shape_measures,
    )

    df, name = _load_dataframe(dataset, source)

    summary = describe_dataframe(df)

    console.print(f"\n[bold cyan]📊 Resumo do dataset: {name}[/bold cyan]")
    console.print(f"  Shape : [yellow]{summary['shape'][0]} linhas × {summary['shape'][1]} colunas[/yellow]")
    console.print(f"  Duplicatas : [yellow]{summary['duplicates']}[/yellow]")

    # Null counts table
    null_table = Table(title="Valores Nulos por Coluna", show_header=True, header_style="bold magenta")
    null_table.add_column("Coluna")
    null_table.add_column("Nulos", justify="right")
    null_table.add_column("% Nulos", justify="right")
    for col, cnt in summary["null_counts"].items():
        pct = summary["null_pct"][col]
        null_table.add_row(col, str(cnt), f"{pct:.2f}%")
    console.print(null_table)

    # Per-column stats for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        stat_table = Table(title="Estatísticas por Coluna Numérica", show_header=True, header_style="bold magenta")
        stat_table.add_column("Coluna")
        stat_table.add_column("Média", justify="right")
        stat_table.add_column("Mediana", justify="right")
        stat_table.add_column("Desvio Padrão", justify="right")
        stat_table.add_column("Skewness", justify="right")
        stat_table.add_column("Outliers (IQR)", justify="right")
        for col in numeric_cols:
            ct = central_tendency(df[col])
            disp = dispersion(df[col])
            shape = shape_measures(df[col])
            outs = detect_outliers(df[col], method="iqr")
            stat_table.add_row(
                col,
                f"{ct['mean']:.4f}",
                f"{ct['median']:.4f}",
                f"{disp['std']:.4f}",
                f"{shape['skewness']:.4f}",
                str(outs["n_outliers"]),
            )
        console.print(stat_table)

    if output:
        from ml_clustering_lab.utils.io import save_json

        report: dict = {
            "dataset": name,
            "shape": list(summary["shape"]),
            "duplicates": summary["duplicates"],
            "null_counts": summary["null_counts"],
            "null_pct": summary["null_pct"],
            "columns": {},
        }
        for col in numeric_cols:
            report["columns"][col] = {
                **central_tendency(df[col]),
                **dispersion(df[col]),
                **shape_measures(df[col]),
                "outliers": detect_outliers(df[col], method="iqr"),
            }
        save_json(report, output)
        console.print(f"\n[green]✔ Relatório salvo em: {output}[/green]")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


@app.command()
def cluster(
    algorithm: str = typer.Option(
        "kmeans",
        "--algorithm",
        "-a",
        help="Algoritmo de clustering: kmeans | dbscan | agglomerative | mean-shift.",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Nome de um dataset embutido (ex.: iris, wine, blobs).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Caminho para um arquivo CSV local.",
    ),
    n_clusters: Optional[int] = typer.Option(
        None,
        "--n-clusters",
        "-k",
        help="Número de clusters (K-Means e Aglomerativo).",
    ),
    eps: Optional[float] = typer.Option(
        None,
        "--eps",
        help="Raio de vizinhança para DBSCAN.",
    ),
    min_samples: Optional[int] = typer.Option(
        None,
        "--min-samples",
        help="Mínimo de amostras por cluster para DBSCAN.",
    ),
    linkage: Optional[str] = typer.Option(
        None,
        "--linkage",
        help="Método de linkage para Aglomerativo: ward | complete | average | single.",
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        help="Diretório para salvar os artefatos do experimento.",
    ),
) -> None:
    """Executa um algoritmo de clustering num dataset.

    Carrega o dataset, aplica pré-processamento padrão (remoção de nulos,
    escalonamento), executa o algoritmo escolhido, calcula métricas de avaliação
    (silhouette, Davies-Bouldin, Calinski-Harabász) e salva os resultados.

    Exemplos::

        ml-lab cluster --dataset iris --algorithm kmeans --n-clusters 3
        ml-lab cluster --dataset wine --algorithm dbscan --eps 0.5 --min-samples 5
        ml-lab cluster --source data/raw/dados.csv --algorithm agglomerative -k 4 --linkage ward
        ml-lab cluster --dataset iris --algorithm mean-shift
    """
    from ml_clustering_lab.pipeline.run_single import run_single

    if dataset is None and source is None:
        console.print("[red]Especifique --dataset ou --source.[/red]")
        raise typer.Exit(code=1)

    algo_kwargs: dict = {}
    if eps is not None:
        algo_kwargs["eps"] = eps
    if min_samples is not None:
        algo_kwargs["min_samples"] = min_samples
    if linkage is not None:
        algo_kwargs["linkage"] = linkage

    console.print(f"[bold cyan]🔬 Executando {algorithm} em '{dataset or source}'...[/bold cyan]")

    try:
        result = run_single(
            algorithm=algorithm,
            dataset=dataset,
            source=source,
            n_clusters=n_clusters,
            outdir=outdir,
            **algo_kwargs,
        )
    except Exception as exc:
        console.print(f"[red]Erro: {exc}[/red]")
        raise typer.Exit(code=1)

    metrics = result["metrics"]

    # Internal metrics table
    met_table = Table(title="Métricas Internas", show_header=True, header_style="bold magenta")
    met_table.add_column("Métrica")
    met_table.add_column("Valor", justify="right")
    met_table.add_row("Clusters encontrados", str(int(metrics.get("n_clusters", 0))))
    met_table.add_row("Pontos de ruído", str(int(metrics.get("n_noise", 0))))
    sil = metrics.get("silhouette", float("nan"))
    met_table.add_row("Silhouette Score", f"{sil:.4f}" if sil == sil else "N/A")
    dbi = metrics.get("davies_bouldin", float("nan"))
    met_table.add_row("Davies-Bouldin Index", f"{dbi:.4f}" if dbi == dbi else "N/A")
    chi = metrics.get("calinski_harabasz", float("nan"))
    met_table.add_row("Calinski-Harabász", f"{chi:.4f}" if chi == chi else "N/A")
    met_table.add_row("Tempo (s)", f"{metrics.get('elapsed_time', 0):.4f}")
    console.print(met_table)

    # External metrics (if available)
    ext = result.get("external_metrics")
    if ext:
        ext_table = Table(title="Métricas Externas (vs. target real)", show_header=True, header_style="bold magenta")
        ext_table.add_column("Métrica")
        ext_table.add_column("Valor", justify="right")
        ext_table.add_row("Adjusted Rand Index", f"{ext.get('adjusted_rand_index', float('nan')):.4f}")
        ext_table.add_row("Normalized Mutual Info", f"{ext.get('normalized_mutual_info', float('nan')):.4f}")
        ext_table.add_row("V-measure", f"{ext.get('v_measure', float('nan')):.4f}")
        console.print(ext_table)

    artifacts = result.get("artifacts", {})
    console.print(f"\n[green]✔ Artefatos salvos em:[/green]")
    for key, path in artifacts.items():
        console.print(f"  {key}: [dim]{path}[/dim]")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Nome de um dataset embutido (ex.: iris, wine, blobs).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Caminho para um arquivo CSV local.",
    ),
    algorithms: str = typer.Option(
        "kmeans,dbscan,agglomerative,mean-shift",
        "--algorithms",
        help="Algoritmos separados por vírgula.",
    ),
    n_clusters: Optional[int] = typer.Option(
        None,
        "--n-clusters",
        "-k",
        help="Número de clusters para K-Means e Aglomerativo.",
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        help="Diretório para salvar os artefatos comparativos.",
    ),
) -> None:
    """Compara múltiplos algoritmos de clustering no mesmo dataset.

    Aplica o mesmo pré-processamento em todos os algoritmos solicitados, executa
    cada um deles, coleta métricas (silhouette, Davies-Bouldin, Calinski-Harabász,
    número de clusters, ruídos, tempo de execução) e gera uma tabela comparativa
    e gráficos lado a lado.

    Exemplos::

        ml-lab compare --dataset iris
        ml-lab compare --dataset wine --algorithms kmeans,dbscan --outdir outputs/runs/wine
    """
    from ml_clustering_lab.pipeline.run_compare import run_compare

    if dataset is None and source is None:
        console.print("[red]Especifique --dataset ou --source.[/red]")
        raise typer.Exit(code=1)

    algo_list = [a.strip() for a in algorithms.split(",") if a.strip()]
    console.print(f"[bold cyan]🔬 Comparando {algo_list} em '{dataset or source}'...[/bold cyan]")

    try:
        comparison_df = run_compare(
            algorithms=algo_list,
            dataset=dataset,
            source=source,
            n_clusters=n_clusters,
            outdir=outdir,
        )
    except Exception as exc:
        console.print(f"[red]Erro: {exc}[/red]")
        raise typer.Exit(code=1)

    cmp_table = Table(title="Comparação de Algoritmos", show_header=True, header_style="bold magenta")
    for col in comparison_df.columns:
        cmp_table.add_column(col, justify="right" if col != "algorithm" else "left")
    for _, row in comparison_df.iterrows():
        cmp_table.add_row(*[
            f"{v:.4f}" if isinstance(v, float) and v == v else ("N/A" if isinstance(v, float) else str(v))
            for v in row
        ])
    console.print(cmp_table)

    if outdir:
        console.print(f"\n[green]✔ Artefatos salvos em: {outdir}[/green]")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


@app.command()
def dataset(
    list_datasets: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="Lista todos os datasets disponíveis.",
    ),
    from_url: Optional[str] = typer.Option(
        None,
        "--from-url",
        help="URL de um arquivo CSV para carregar e inspecionar.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Nome para registrar o dataset carregado de URL.",
    ),
) -> None:
    """Lista datasets disponíveis ou carrega um novo de URL.

    Permite explorar quais datasets embutidos estão disponíveis (scikit-learn,
    sintéticos) e carregar novos datasets a partir de URLs externas.

    Exemplos::

        ml-lab dataset --list
        ml-lab dataset --from-url "https://example.com/data.csv" --name meu_dataset
    """
    from ml_clustering_lab.datasets.registry import DatasetRegistry

    if list_datasets:
        reg = DatasetRegistry()
        ds_table = Table(title="Datasets Disponíveis", show_header=True, header_style="bold magenta")
        ds_table.add_column("Nome")
        ds_table.add_column("Origem")
        ds_table.add_column("Descrição")
        ds_table.add_column("Amostras", justify="right")
        ds_table.add_column("Features", justify="right")
        for ds_name in reg.list_names():
            info = reg.get(ds_name)
            ds_table.add_row(
                info.name,
                info.source,
                info.description,
                str(info.n_samples) if info.n_samples is not None else "—",
                str(info.n_features) if info.n_features is not None else "—",
            )
        console.print(ds_table)
        return

    if from_url:
        from ml_clustering_lab.datasets.loaders import load_from_url

        label = name or from_url
        console.print(f"[bold cyan]⬇ Carregando dataset de {from_url}...[/bold cyan]")
        try:
            df = load_from_url(from_url)
        except Exception as exc:
            console.print(f"[red]Erro ao carregar URL: {exc}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[green]✔ Carregado: {df.shape[0]} linhas × {df.shape[1]} colunas[/green]")
        console.print(f"[dim]Colunas: {list(df.columns)}[/dim]")
        console.print(Panel(str(df.head(5)), title=f"Head — {label}", border_style="dim"))
        return

    console.print("[yellow]Use --list para listar os datasets ou --from-url para carregar de uma URL.[/yellow]")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


@app.command()
def plot(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Nome de um dataset embutido (ex.: iris, wine, blobs).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Caminho para um arquivo CSV local.",
    ),
    plot_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Tipo de gráfico: histogram | boxplot | correlation | all.",
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Diretório para salvar os gráficos gerados.",
    ),
) -> None:
    """Gera visualizações exploratórias para um dataset.

    Produz gráficos univariados (histogramas, boxplots) e multivariados
    (heatmap de correlação), salvando-os no diretório especificado em formato PNG.

    Exemplos::

        ml-lab plot --dataset iris --outdir outputs/figures/iris
        ml-lab plot --dataset iris --type correlation
        ml-lab plot --source data/raw/dados.csv --type histogram
    """
    from ml_clustering_lab.visualization.plots import (
        plot_boxplot,
        plot_correlation,
        plot_histogram,
    )

    if dataset is None and source is None:
        console.print("[red]Especifique --dataset ou --source.[/red]")
        raise typer.Exit(code=1)

    df, ds_name = _load_dataframe(dataset, source)
    out = outdir or f"outputs/figures/{ds_name}"

    kind = (plot_type or "all").lower()
    _ALL_TYPES = {"histogram", "boxplot", "correlation"}
    if kind not in _ALL_TYPES and kind != "all":
        console.print(f"[red]Tipo de gráfico '{kind}' não suportado. Use: {sorted(_ALL_TYPES)} ou 'all'.[/red]")
        raise typer.Exit(code=1)

    generated: list[str] = []

    if kind in {"histogram", "all"}:
        plot_histogram(df, outdir=out)
        generated.append("histogram.png")

    if kind in {"boxplot", "all"}:
        plot_boxplot(df, outdir=out)
        generated.append("boxplot.png")

    if kind in {"correlation", "all"}:
        plot_correlation(df, outdir=out)
        generated.append("correlation.png")

    console.print(f"[green]✔ Gráfico(s) salvos em: {out}[/green]")
    for fname in generated:
        console.print(f"  [dim]{out}/{fname}[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()


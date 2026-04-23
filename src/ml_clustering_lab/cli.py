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
# Helpers
# ---------------------------------------------------------------------------


def _not_implemented_warning(command: str) -> None:
    """Exibe aviso amigável de que um comando ainda não foi implementado."""
    console.print(
        Panel(
            f"[yellow]O comando [bold]{command}[/bold] ainda não foi implementado.\n"
            "Esta é a estrutura inicial do projeto. "
            "Veja o README para o roadmap de implementação.[/yellow]",
            title="[bold red]Em construção[/bold red]",
            border_style="yellow",
        )
    )


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
    _not_implemented_warning("stats")
    if dataset:
        console.print(f"[dim]Dataset solicitado: {dataset}[/dim]")
    if source:
        console.print(f"[dim]Fonte: {source}[/dim]")
    if output:
        console.print(f"[dim]Saída: {output}[/dim]")


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
    _not_implemented_warning("cluster")
    console.print(f"[dim]Algoritmo: {algorithm}[/dim]")
    if dataset:
        console.print(f"[dim]Dataset: {dataset}[/dim]")


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
    _not_implemented_warning("compare")
    algo_list = [a.strip() for a in algorithms.split(",")]
    console.print(f"[dim]Algoritmos a comparar: {algo_list}[/dim]")
    if dataset:
        console.print(f"[dim]Dataset: {dataset}[/dim]")


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
        help="URL de um arquivo CSV para baixar e registrar.",
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
    _not_implemented_warning("dataset")
    if list_datasets:
        console.print("[dim]Listando datasets disponíveis...[/dim]")
    if from_url:
        console.print(f"[dim]URL: {from_url} → nome: {name}[/dim]")


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
        help="Tipo de gráfico: histogram | boxplot | correlation | scatter | pairplot.",
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Diretório para salvar os gráficos gerados.",
    ),
) -> None:
    """Gera visualizações exploratórias para um dataset.

    Produz gráficos univariados (histogramas, boxplots), bivariados (scatter,
    pairplot) e multivariados (heatmap de correlação), salvando-os no diretório
    especificado em formato PNG.

    Exemplos::

        ml-lab plot --dataset iris --outdir outputs/figures/iris
        ml-lab plot --dataset iris --type correlation
        ml-lab plot --source data/raw/dados.csv --type pairplot
    """
    _not_implemented_warning("plot")
    if dataset:
        console.print(f"[dim]Dataset: {dataset}[/dim]")
    if plot_type:
        console.print(f"[dim]Tipo de gráfico: {plot_type}[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

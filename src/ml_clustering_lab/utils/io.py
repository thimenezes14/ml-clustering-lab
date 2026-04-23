"""
utils/io.py
===========
Funções utilitárias de entrada/saída para artefatos do projeto.

Responsabilidade
----------------
Centralizar operações de IO comuns:
- Garantir a existência de diretórios
- Salvar e carregar arquivos JSON (métricas, configurações)
- Salvar DataFrames como CSV
- Salvar figuras matplotlib como PNG

Extensão futura
---------------
- Suporte a Parquet para artefatos tabulares
- Compressão automática de artefatos antigos
- Versionamento de artefatos por timestamp
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Garante que um diretório existe, criando-o se necessário.

    Parâmetros
    ----------
    path : str | Path
        Caminho do diretório.

    Retorna
    -------
    Path
        Objeto Path do diretório criado/existente.

    Exemplo
    -------
    >>> dir_path = ensure_dir("outputs/figures/iris")
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Salva um dicionário como arquivo JSON.

    Parâmetros
    ----------
    data : dict
        Dados a serializar.
    path : str | Path
        Caminho de destino do arquivo.
    indent : int, default=2
        Indentação para formatação legível.

    Exemplo
    -------
    >>> save_json(metrics, "outputs/reports/metrics.json")
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict:
    """Carrega um arquivo JSON como dicionário.

    Parâmetros
    ----------
    path : str | Path
        Caminho do arquivo JSON.

    Retorna
    -------
    dict
        Conteúdo deserializado.

    Exceções
    --------
    FileNotFoundError
        Se o arquivo não existir.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Salva um DataFrame como arquivo CSV.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a salvar.
    path : str | Path
        Caminho de destino (deve ter extensão .csv).

    Exemplo
    -------
    >>> save_dataframe(comparison_df, "outputs/reports/comparison.csv")
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def save_figure(fig, path: str | Path, dpi: int = 150) -> None:
    """Salva uma figura matplotlib como PNG.

    Parâmetros
    ----------
    fig : matplotlib.figure.Figure
        Figura a salvar.
    path : str | Path
        Caminho de destino (deve ter extensão .png).
    dpi : int, default=150
        Resolução da imagem em pontos por polegada.

    Exemplo
    -------
    >>> save_figure(fig, "outputs/figures/iris/histogram.png")
    """
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")

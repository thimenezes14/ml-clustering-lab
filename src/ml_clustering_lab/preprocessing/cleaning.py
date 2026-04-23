"""
preprocessing/cleaning.py
==========================
Funções de limpeza e tratamento básico de um DataFrame.

Responsabilidade
----------------
Garantir a qualidade mínima dos dados antes das etapas de escalonamento
e clustering: remoção de nulos, duplicados e outliers extremos.

Extensão futura
---------------
- Estratégias de imputação configuráveis (média, mediana, KNN)
- Detecção e correção de tipos de coluna
- Suporte a pipelines sklearn via ``BaseEstimator`` / ``TransformerMixin``
"""

from __future__ import annotations

import pandas as pd


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Remove linhas e colunas com excesso de valores ausentes.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    threshold : float, default=0.5
        Fração máxima permitida de valores nulos por coluna.
        Colunas com proporção de nulos acima deste valor são removidas primeiro;
        em seguida, linhas restantes com qualquer nulo são descartadas.

    Retorna
    -------
    pd.DataFrame
        DataFrame sem valores ausentes.

    Exemplo
    -------
    >>> clean_df = drop_missing(df, threshold=0.3)

    Extensão futura
    ---------------
    - Parâmetro ``strategy`` para imputação em vez de remoção
    - Log detalhado das colunas e linhas removidas
    """
    raise NotImplementedError("drop_missing ainda não foi implementado.")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas do DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame sem linhas duplicadas.

    Exemplo
    -------
    >>> clean_df = drop_duplicates(df)

    Extensão futura
    ---------------
    - Parâmetro ``subset`` para considerar apenas certas colunas
    - Estatísticas sobre o número de duplicatas removidas
    """
    raise NotImplementedError("drop_duplicates ainda não foi implementado.")


def remove_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
    """Remove ou marca outliers nas colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada (apenas colunas numéricas são avaliadas).
    method : str, default="iqr"
        Método de detecção: ``"iqr"`` (intervalo interquartil) ou ``"zscore"``.
    threshold : float, default=3.0
        Multiplicador do IQR ou limiar de z-score para identificar outliers.

    Retorna
    -------
    pd.DataFrame
        DataFrame com linhas contendo outliers removidas.

    Exemplo
    -------
    >>> clean_df = remove_outliers(df, method="zscore", threshold=2.5)

    Extensão futura
    ---------------
    - Parâmetro ``action`` para escolher entre remoção, clipping ou marcação
    - Método ``isolation_forest`` para detecção multivariada
    """
    raise NotImplementedError("remove_outliers ainda não foi implementado.")

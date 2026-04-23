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
    # Drop columns where the proportion of nulls exceeds the threshold
    col_null_frac = df.isnull().mean()
    cols_to_keep = col_null_frac[col_null_frac <= threshold].index.tolist()
    df = df[cols_to_keep]
    # Drop remaining rows with any null
    return df.dropna().reset_index(drop=True)


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
    return df.drop_duplicates().reset_index(drop=True)


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
    import numpy as np

    numeric_cols = df.select_dtypes(include="number").columns
    mask = pd.Series(True, index=df.index)

    for col in numeric_cols:
        series = df[col].dropna()
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            raise ValueError(f"Método '{method}' não suportado. Use 'iqr' ou 'zscore'.")
        mask &= df[col].between(lower, upper)

    return df[mask].reset_index(drop=True)

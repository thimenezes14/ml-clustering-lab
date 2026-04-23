"""
preprocessing/feature_selection.py
====================================
Funções de seleção de features para uso nos algoritmos de clustering.

Responsabilidade
----------------
Selecionar ou filtrar as colunas relevantes do DataFrame antes do clustering.
O foco inicial é na seleção de features numéricas, excluindo colunas não
informativas como identificadores, colunas constantes e colunas alvo.

Extensão futura
---------------
- Seleção baseada em variância (``VarianceThreshold``)
- Seleção baseada em correlação (remover features redundantes)
- PCA como passo de redução de dimensionalidade antes do clustering
- Seleção guiada por importância de features (Random Forest, etc.)
"""

from __future__ import annotations

import pandas as pd


def select_numeric_features(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Seleciona apenas as colunas numéricas do DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    exclude : list[str] | None, default=None
        Nomes de colunas a excluir mesmo que numéricas
        (ex.: colunas de ID, coluna ``target``).

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo apenas as colunas numéricas relevantes.

    Exemplo
    -------
    >>> X = select_numeric_features(df, exclude=["target", "id"])

    Extensão futura
    ---------------
    - Parâmetro ``min_variance`` para remover colunas quase constantes
    - Parâmetro ``max_correlation`` para remover features altamente correlacionadas
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if exclude:
        numeric_cols = [c for c in numeric_cols if c not in exclude]
    return df[numeric_cols].copy()

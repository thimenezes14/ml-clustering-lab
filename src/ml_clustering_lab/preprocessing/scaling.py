"""
preprocessing/scaling.py
=========================
Funções de normalização e padronização de features numéricas.

Responsabilidade
----------------
Aplicar escalonamento consistente nas features antes do clustering, pois a maioria
dos algoritmos (K-Means, DBSCAN, Aglomerativo) é sensível à escala dos dados.

Escaladores disponíveis (planejados)
--------------------------------------
- ``StandardScaler``  : z-score (média 0, desvio 1) — padrão para K-Means
- ``MinMaxScaler``    : escala para [0, 1]
- ``RobustScaler``    : robusto a outliers, usa mediana e IQR

Extensão futura
---------------
- Retornar o scaler fitado para reutilização (inverter transformação)
- Suporte a pipeline sklearn
- Logging da escala aplicada por coluna
"""

from __future__ import annotations

import pandas as pd


def scale_features(
    df: pd.DataFrame,
    method: str = "standard",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Escala as features numéricas de um DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    method : str, default="standard"
        Método de escalonamento: ``"standard"`` | ``"minmax"`` | ``"robust"``.
    columns : list[str] | None, default=None
        Colunas a escalar. Se None, todas as colunas numéricas são escaladas.

    Retorna
    -------
    pd.DataFrame
        DataFrame com as colunas escaladas. Colunas não numéricas são mantidas
        inalteradas.

    Exceções
    --------
    ValueError
        Se ``method`` não for um dos valores suportados.

    Exemplo
    -------
    >>> scaled_df = scale_features(df, method="standard")

    Extensão futura
    ---------------
    - Retornar tupla ``(scaled_df, scaler)`` para reutilização
    - Suporte a ``QuantileTransformer`` e ``PowerTransformer``
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    _SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in _SCALERS:
        raise ValueError(f"Método '{method}' não suportado. Use: {list(_SCALERS.keys())}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if columns is not None:
        # Use only requested columns that exist and are numeric
        numeric_cols = [c for c in columns if c in numeric_cols]

    if not numeric_cols:
        return df.copy()

    scaler = _SCALERS[method]()
    result = df.copy()
    result[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return result

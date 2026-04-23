"""
stats/descriptive.py
=====================
Funções de análise estatística descritiva.

Responsabilidade
----------------
Calcular e retornar métricas estatísticas descritivas para séries e DataFrames:

- **Tendência central** : média, mediana, moda
- **Dispersão**         : variância, desvio padrão, amplitude, IQR, coeficiente de variação
- **Posição**           : quartis, percentis configuráveis
- **Forma**             : assimetria (skewness) e curtose (kurtosis)
- **Qualidade**         : nulos, duplicados, outliers, tipos de colunas

Todas as funções recebem ``pandas.Series`` ou ``pandas.DataFrame`` e retornam
estruturas nativas Python ou ``pandas``, tornando fácil exibir ou serializar.

Extensão futura
---------------
- Suporte a intervalos de confiança (bootstrap)
- Testes de normalidade (Shapiro-Wilk, D'Agostino-Pearson)
- Sumário automático comparando distribuições antes e após pré-processamento
"""

from __future__ import annotations

import pandas as pd


def describe_dataframe(df: pd.DataFrame) -> dict:
    """Gera um resumo completo do DataFrame.

    Inclui: shape, tipos de colunas, contagem de nulos por coluna, porcentagem
    de nulos, duplicatas, cardinalidade e um resumo estatístico de todas as
    colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    dict
        Dicionário com chaves:
        - ``shape``         : tuple (n_rows, n_cols)
        - ``dtypes``        : dict coluna → dtype
        - ``null_counts``   : dict coluna → contagem de nulos
        - ``null_pct``      : dict coluna → porcentagem de nulos
        - ``duplicates``    : int (linhas duplicadas)
        - ``numeric_summary``: DataFrame com estatísticas das colunas numéricas

    Exemplo
    -------
    >>> summary = describe_dataframe(df)
    >>> summary["shape"]
    (150, 5)

    Extensão futura
    ---------------
    - Exportar para Markdown ou HTML
    - Incluir histogramas inline (usando ``rich`` ou ``plotext``)
    """
    numeric_cols = df.select_dtypes(include="number")
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": numeric_cols.describe().T,
    }


def central_tendency(series: pd.Series) -> dict[str, float]:
    """Calcula medidas de tendência central de uma série numérica.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.

    Retorna
    -------
    dict[str, float]
        Dicionário com:
        - ``mean``   : média aritmética
        - ``median`` : mediana
        - ``mode``   : moda (primeiro valor se multimodal)

    Exemplo
    -------
    >>> central_tendency(df["sepal_length"])
    {'mean': 5.843, 'median': 5.8, 'mode': 5.0}

    Extensão futura
    ---------------
    - Incluir média aparada (trimmed mean)
    - Indicar se a distribuição é multimodal
    """
    mode_vals = series.mode()
    mode_val = float(mode_vals.iloc[0]) if not mode_vals.empty else float("nan")
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "mode": mode_val,
    }


def dispersion(series: pd.Series) -> dict[str, float]:
    """Calcula medidas de dispersão de uma série numérica.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.

    Retorna
    -------
    dict[str, float]
        Dicionário com:
        - ``variance``  : variância amostral
        - ``std``       : desvio padrão amostral
        - ``range``     : amplitude (max - min)
        - ``iqr``       : intervalo interquartil (Q3 - Q1)
        - ``cv``        : coeficiente de variação (std / mean * 100)

    Exemplo
    -------
    >>> dispersion(df["sepal_length"])
    {'variance': 0.686, 'std': 0.828, 'range': 3.6, 'iqr': 1.3, 'cv': 14.17}

    Extensão futura
    ---------------
    - Incluir desvio mediano absoluto (MAD)
    - Incluir intervalo de confiança para a variância
    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    mean = float(series.mean())
    std = float(series.std())
    cv = (std / mean * 100) if mean != 0 else float("nan")
    return {
        "variance": float(series.var()),
        "std": std,
        "range": float(series.max() - series.min()),
        "iqr": round(q3 - q1, 6),
        "cv": round(cv, 6),
    }


def position_measures(
    series: pd.Series,
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Calcula medidas de posição de uma série numérica.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.
    percentiles : list[float] | None, default=None
        Lista de percentis a calcular (entre 0 e 1).
        Se None, usa [0.10, 0.25, 0.50, 0.75, 0.90].

    Retorna
    -------
    dict[str, float]
        Dicionário com chaves ``p{N}`` para cada percentil solicitado
        (ex.: ``p25``, ``p50``, ``p75``) e chaves ``q1``, ``q2``, ``q3``.

    Exemplo
    -------
    >>> position_measures(df["sepal_length"], percentiles=[0.25, 0.5, 0.75])
    {'q1': 5.1, 'q2': 5.8, 'q3': 6.4, 'p25': 5.1, 'p50': 5.8, 'p75': 6.4}

    Extensão futura
    ---------------
    - Incluir cálculo de decis
    - Visualização de box com percentis
    """
    if percentiles is None:
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    result: dict[str, float] = {}
    for p in percentiles:
        key = f"p{int(round(p * 100))}"
        result[key] = float(series.quantile(p))

    result["q1"] = float(series.quantile(0.25))
    result["q2"] = float(series.quantile(0.50))
    result["q3"] = float(series.quantile(0.75))
    return result


def shape_measures(series: pd.Series) -> dict[str, float]:
    """Calcula medidas de forma da distribuição de uma série numérica.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.

    Retorna
    -------
    dict[str, float]
        Dicionário com:
        - ``skewness`` : assimetria (0 = simétrica, >0 = cauda direita)
        - ``kurtosis`` : curtose (0 = normal, >0 = leptocúrtica, <0 = platicúrtica)

    Exemplo
    -------
    >>> shape_measures(df["sepal_length"])
    {'skewness': 0.314, 'kurtosis': -0.574}

    Extensão futura
    ---------------
    - Incluir resultado de teste de normalidade (Shapiro-Wilk p-value)
    - Classificar forma automaticamente (simétrica, assimétrica, etc.)
    """
    return {
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
    }


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 3.0,
) -> dict:
    """Detecta outliers em uma série numérica.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.
    method : str, default="iqr"
        Método de detecção: ``"iqr"`` ou ``"zscore"``.
    threshold : float, default=3.0
        Multiplicador do IQR (método ``iqr``) ou limite de z-score (método ``zscore``).

    Retorna
    -------
    dict
        Dicionário com:
        - ``n_outliers``    : int — número de outliers detectados
        - ``pct_outliers``  : float — porcentagem de outliers
        - ``lower_bound``   : float — limite inferior
        - ``upper_bound``   : float — limite superior
        - ``outlier_indices``: list[int] — índices das linhas com outliers

    Exemplo
    -------
    >>> detect_outliers(df["sepal_length"], method="iqr")
    {'n_outliers': 0, 'pct_outliers': 0.0, ...}

    Extensão futura
    ---------------
    - Método ``isolation_forest`` para detecção multivariada
    - Visualização automática dos outliers no boxplot
    """
    import numpy as np

    clean = series.dropna()

    if method == "iqr":
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == "zscore":
        mean = float(clean.mean())
        std = float(clean.std())
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:
        raise ValueError(f"Método '{method}' não suportado. Use 'iqr' ou 'zscore'.")

    outlier_mask = (clean < lower) | (clean > upper)
    outlier_indices = clean.index[outlier_mask].tolist()
    n = int(outlier_mask.sum())
    pct = round(n / len(clean) * 100, 4) if len(clean) > 0 else 0.0

    return {
        "n_outliers": n,
        "pct_outliers": pct,
        "lower_bound": round(lower, 6),
        "upper_bound": round(upper, 6),
        "outlier_indices": outlier_indices,
    }

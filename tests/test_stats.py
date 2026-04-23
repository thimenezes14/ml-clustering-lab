"""
tests/test_stats.py
====================
Testes para o módulo de análise estatística descritiva
(ml_clustering_lab.stats.descriptive).

Cobertura planejada
-------------------
- describe_dataframe: shape, dtypes, nulos, duplicados
- central_tendency: média, mediana, moda em séries simples
- dispersion: variância, desvio padrão, IQR, amplitude, CV
- position_measures: quartis e percentis customizados
- shape_measures: skewness e kurtosis
- detect_outliers: método IQR e z-score

Nota
----
Todos os testes estão como stubs (``NotImplementedError``) enquanto as
funções não estiverem implementadas. Atualize-os conforme a implementação
avançar.
"""

import pandas as pd
import pytest

from ml_clustering_lab.stats.descriptive import (
    central_tendency,
    describe_dataframe,
    detect_outliers,
    dispersion,
    position_measures,
    shape_measures,
)


@pytest.fixture
def simple_series() -> pd.Series:
    """Série numérica simples para testes."""
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """DataFrame simples com duas colunas numéricas."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


class TestDescribeDataframe:
    def test_raises_not_implemented(self, simple_df):
        with pytest.raises(NotImplementedError):
            describe_dataframe(simple_df)


class TestCentralTendency:
    def test_raises_not_implemented(self, simple_series):
        with pytest.raises(NotImplementedError):
            central_tendency(simple_series)


class TestDispersion:
    def test_raises_not_implemented(self, simple_series):
        with pytest.raises(NotImplementedError):
            dispersion(simple_series)


class TestPositionMeasures:
    def test_raises_not_implemented(self, simple_series):
        with pytest.raises(NotImplementedError):
            position_measures(simple_series)


class TestShapeMeasures:
    def test_raises_not_implemented(self, simple_series):
        with pytest.raises(NotImplementedError):
            shape_measures(simple_series)


class TestDetectOutliers:
    def test_raises_not_implemented_iqr(self, simple_series):
        with pytest.raises(NotImplementedError):
            detect_outliers(simple_series, method="iqr")

    def test_raises_not_implemented_zscore(self, simple_series):
        with pytest.raises(NotImplementedError):
            detect_outliers(simple_series, method="zscore")

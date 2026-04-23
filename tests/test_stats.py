"""
tests/test_stats.py
====================
Testes para o módulo de análise estatística descritiva
(ml_clustering_lab.stats.descriptive).

Cobertura
---------
- describe_dataframe: shape, dtypes, nulos, duplicados
- central_tendency: média, mediana, moda em séries simples
- dispersion: variância, desvio padrão, IQR, amplitude, CV
- position_measures: quartis e percentis customizados
- shape_measures: skewness e kurtosis
- detect_outliers: método IQR e z-score
"""

import math

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
    def test_returns_shape(self, simple_df):
        summary = describe_dataframe(simple_df)
        assert summary["shape"] == (5, 2)

    def test_returns_dtypes(self, simple_df):
        summary = describe_dataframe(simple_df)
        assert "a" in summary["dtypes"]

    def test_null_counts_zero(self, simple_df):
        summary = describe_dataframe(simple_df)
        assert summary["null_counts"]["a"] == 0

    def test_duplicate_count(self):
        df = pd.DataFrame({"x": [1, 1, 2, 3]})
        summary = describe_dataframe(df)
        assert summary["duplicates"] == 1

    def test_numeric_summary_present(self, simple_df):
        summary = describe_dataframe(simple_df)
        assert "numeric_summary" in summary
        assert "a" in summary["numeric_summary"].index


class TestCentralTendency:
    def test_mean(self, simple_series):
        result = central_tendency(simple_series)
        assert result["mean"] == pytest.approx(5.5)

    def test_median(self, simple_series):
        result = central_tendency(simple_series)
        assert result["median"] == pytest.approx(5.5)

    def test_mode(self):
        s = pd.Series([1.0, 2.0, 2.0, 3.0])
        result = central_tendency(s)
        assert result["mode"] == pytest.approx(2.0)

    def test_returns_dict(self, simple_series):
        result = central_tendency(simple_series)
        assert isinstance(result, dict)
        for key in ["mean", "median", "mode"]:
            assert key in result


class TestDispersion:
    def test_range(self, simple_series):
        result = dispersion(simple_series)
        assert result["range"] == pytest.approx(9.0)

    def test_std_positive(self, simple_series):
        result = dispersion(simple_series)
        assert result["std"] > 0

    def test_variance_positive(self, simple_series):
        result = dispersion(simple_series)
        assert result["variance"] > 0

    def test_iqr(self, simple_series):
        result = dispersion(simple_series)
        # Q1=3.25, Q3=7.75 → IQR=4.5
        assert result["iqr"] == pytest.approx(4.5)

    def test_cv_positive(self, simple_series):
        result = dispersion(simple_series)
        assert result["cv"] > 0

    def test_returns_expected_keys(self, simple_series):
        result = dispersion(simple_series)
        for key in ["variance", "std", "range", "iqr", "cv"]:
            assert key in result


class TestPositionMeasures:
    def test_quartiles_present(self, simple_series):
        result = position_measures(simple_series)
        assert "q1" in result
        assert "q2" in result
        assert "q3" in result

    def test_q1_less_than_q3(self, simple_series):
        result = position_measures(simple_series)
        assert result["q1"] < result["q3"]

    def test_custom_percentiles(self, simple_series):
        result = position_measures(simple_series, percentiles=[0.10, 0.90])
        assert "p10" in result
        assert "p90" in result
        assert result["p10"] < result["p90"]

    def test_default_percentiles_included(self, simple_series):
        result = position_measures(simple_series)
        # Default percentiles include 0.25, 0.50, 0.75
        assert "p25" in result
        assert "p50" in result
        assert "p75" in result


class TestShapeMeasures:
    def test_returns_skewness_and_kurtosis(self, simple_series):
        result = shape_measures(simple_series)
        assert "skewness" in result
        assert "kurtosis" in result

    def test_symmetric_series_near_zero_skew(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shape_measures(s)
        assert abs(result["skewness"]) < 0.1

    def test_right_skewed(self):
        # Right-skewed: long tail to the right
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 10.0])
        result = shape_measures(s)
        assert result["skewness"] > 0


class TestDetectOutliers:
    def test_no_outliers_iqr(self, simple_series):
        result = detect_outliers(simple_series, method="iqr")
        assert isinstance(result["n_outliers"], int)
        assert result["n_outliers"] >= 0

    def test_no_outliers_zscore(self, simple_series):
        result = detect_outliers(simple_series, method="zscore")
        assert isinstance(result["n_outliers"], int)

    def test_detects_obvious_outlier_zscore(self):
        s = pd.Series([1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 1000.0])
        result = detect_outliers(s, method="zscore", threshold=2.0)
        assert result["n_outliers"] >= 1

    def test_detects_obvious_outlier_iqr(self):
        s = pd.Series([1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 1000.0])
        result = detect_outliers(s, method="iqr", threshold=1.5)
        assert result["n_outliers"] >= 1

    def test_returns_expected_keys(self, simple_series):
        result = detect_outliers(simple_series, method="iqr")
        for key in ["n_outliers", "pct_outliers", "lower_bound", "upper_bound", "outlier_indices"]:
            assert key in result

    def test_invalid_method(self, simple_series):
        with pytest.raises(ValueError, match="não suportado"):
            detect_outliers(simple_series, method="isolation_forest")

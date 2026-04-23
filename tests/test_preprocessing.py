"""
tests/test_preprocessing.py
============================
Testes para os módulos de pré-processamento
(ml_clustering_lab.preprocessing).

Cobertura
---------
- cleaning: drop_missing, drop_duplicates, remove_outliers
- feature_selection: select_numeric_features
- scaling: scale_features
"""

import numpy as np
import pandas as pd
import pytest

from ml_clustering_lab.preprocessing.cleaning import (
    drop_duplicates,
    drop_missing,
    remove_outliers,
)
from ml_clustering_lab.preprocessing.feature_selection import select_numeric_features
from ml_clustering_lab.preprocessing.scaling import scale_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, None, 40.0, 50.0],
        }
    )


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 2.0, 3.0],
            "b": [10.0, 20.0, 20.0, 30.0],
        }
    )


# ---------------------------------------------------------------------------
# drop_missing
# ---------------------------------------------------------------------------


class TestDropMissing:
    def test_removes_rows_with_nulls(self, df_with_nulls):
        clean = drop_missing(df_with_nulls)
        assert clean.isnull().sum().sum() == 0

    def test_no_nulls_unchanged(self, simple_df):
        clean = drop_missing(simple_df)
        assert clean.shape == simple_df.shape

    def test_returns_dataframe(self, df_with_nulls):
        result = drop_missing(df_with_nulls)
        assert isinstance(result, pd.DataFrame)

    def test_index_reset(self, df_with_nulls):
        clean = drop_missing(df_with_nulls)
        assert list(clean.index) == list(range(len(clean)))

    def test_drops_column_with_high_null_fraction(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [None, None, None, None, 1.0],  # 80% nulls
            }
        )
        clean = drop_missing(df, threshold=0.5)
        assert "b" not in clean.columns


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    def test_removes_duplicates(self, df_with_duplicates):
        clean = drop_duplicates(df_with_duplicates)
        assert len(clean) == 3

    def test_no_duplicates_unchanged(self, simple_df):
        clean = drop_duplicates(simple_df)
        assert clean.shape == simple_df.shape

    def test_returns_dataframe(self, df_with_duplicates):
        result = drop_duplicates(df_with_duplicates)
        assert isinstance(result, pd.DataFrame)

    def test_index_reset(self, df_with_duplicates):
        clean = drop_duplicates(df_with_duplicates)
        assert list(clean.index) == list(range(len(clean)))


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------


class TestRemoveOutliers:
    def test_removes_extreme_values_iqr(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 1000.0]})
        clean = remove_outliers(df, method="iqr", threshold=1.5)
        assert 1000.0 not in clean["x"].values

    def test_removes_extreme_values_zscore(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 1000.0]})
        clean = remove_outliers(df, method="zscore", threshold=2.0)
        assert 1000.0 not in clean["x"].values

    def test_no_outliers_unchanged(self, simple_df):
        clean = remove_outliers(simple_df, method="iqr", threshold=3.0)
        assert len(clean) == len(simple_df)

    def test_returns_dataframe(self, simple_df):
        result = remove_outliers(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_method_raises(self, simple_df):
        with pytest.raises(ValueError, match="não suportado"):
            remove_outliers(simple_df, method="invalid")

    def test_index_reset(self, simple_df):
        clean = remove_outliers(simple_df)
        assert list(clean.index) == list(range(len(clean)))


# ---------------------------------------------------------------------------
# select_numeric_features
# ---------------------------------------------------------------------------


class TestSelectNumericFeatures:
    def test_selects_only_numeric_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3.0, 4.0]})
        result = select_numeric_features(df)
        assert list(result.columns) == ["a", "c"]

    def test_excludes_specified_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "target": [0, 1], "c": [3.0, 4.0]})
        result = select_numeric_features(df, exclude=["target"])
        assert "target" not in result.columns
        assert "a" in result.columns

    def test_returns_dataframe(self, simple_df):
        result = select_numeric_features(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_numeric_included_by_default(self, simple_df):
        result = select_numeric_features(simple_df)
        assert set(result.columns) == {"a", "b"}

    def test_exclude_nonexistent_column_is_safe(self, simple_df):
        result = select_numeric_features(simple_df, exclude=["nonexistent"])
        assert set(result.columns) == {"a", "b"}


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    def test_standard_scaler_mean_near_zero(self, simple_df):
        result = scale_features(simple_df, method="standard")
        assert abs(result["a"].mean()) < 1e-10

    def test_standard_scaler_std_near_one(self, simple_df):
        result = scale_features(simple_df, method="standard")
        assert abs(result["a"].std(ddof=0) - 1.0) < 0.1

    def test_minmax_range(self, simple_df):
        result = scale_features(simple_df, method="minmax")
        assert result["a"].min() == pytest.approx(0.0)
        assert result["a"].max() == pytest.approx(1.0)

    def test_robust_scaler_returns_dataframe(self, simple_df):
        result = scale_features(simple_df, method="robust")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_method_raises(self, simple_df):
        with pytest.raises(ValueError, match="não suportado"):
            scale_features(simple_df, method="invalid")

    def test_returns_dataframe(self, simple_df):
        result = scale_features(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_non_numeric_columns_preserved(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]})
        result = scale_features(df)
        assert "label" in result.columns
        assert list(result["label"]) == ["x", "y", "z"]

    def test_scale_specific_columns(self, simple_df):
        result = scale_features(simple_df, columns=["a"])
        # 'b' should remain unchanged
        assert list(result["b"]) == list(simple_df["b"])

    def test_empty_numeric_returns_copy(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = scale_features(df)
        pd.testing.assert_frame_equal(result, df)

"""
preprocessing/__init__.py
==========================
Submódulo de pré-processamento de dados.

Responsabilidade
----------------
Encapsular todas as transformações aplicadas ao dataset antes de ser entregue
aos algoritmos de clustering: limpeza de nulos e duplicados, escalonamento de
features numéricas e seleção das colunas relevantes.

Extensão futura
---------------
- Pipeline sklearn-compatível (``BaseEstimator``, ``TransformerMixin``)
- Encoding de variáveis categóricas
- Feature engineering automático
"""

from ml_clustering_lab.preprocessing.cleaning import (
    drop_duplicates,
    drop_missing,
    remove_outliers,
)
from ml_clustering_lab.preprocessing.feature_selection import select_numeric_features
from ml_clustering_lab.preprocessing.scaling import scale_features

__all__ = [
    "drop_duplicates",
    "drop_missing",
    "remove_outliers",
    "select_numeric_features",
    "scale_features",
]

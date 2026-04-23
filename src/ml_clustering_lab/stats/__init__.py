"""
stats/__init__.py
=================
Submódulo de análise estatística descritiva.

Responsabilidade
----------------
Fornecer medidas de tendência central, dispersão, posição e forma de
distribuição, além de diagnósticos de qualidade de dados (nulos, duplicados,
outliers). Este módulo é independente dos algoritmos de clustering e pode ser
usado de forma autônoma.

Extensão futura
---------------
- Testes estatísticos (normalidade, homogeneidade de variâncias)
- Análise bivariada (correlação de Spearman, Pearson, Cramér's V)
- Sumário automático em HTML/Markdown
"""

from ml_clustering_lab.stats.descriptive import (
    central_tendency,
    describe_dataframe,
    detect_outliers,
    dispersion,
    position_measures,
    shape_measures,
)

__all__ = [
    "describe_dataframe",
    "central_tendency",
    "dispersion",
    "position_measures",
    "shape_measures",
    "detect_outliers",
]

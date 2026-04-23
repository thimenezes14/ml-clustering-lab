"""
stats/distributions.py
=======================
Análise de distribuições estatísticas das features.

Responsabilidade
----------------
Gerar métricas e visualizações relacionadas às distribuições de cada coluna
numérica: histogramas, KDE, testes de normalidade e detecção do tipo de
distribuição (uniforme, normal, bimodal, etc.).

Este módulo complementa ``descriptive.py`` com foco na **forma** das distribuições,
o que é útil para:
- escolher o escalonador adequado
- identificar features com distribuição não normal (ex.: log-transform)
- avaliar a separabilidade antes do clustering

Extensão futura
---------------
- Ajuste automático de distribuição (scipy.stats.fit)
- Testes formais de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov, D'Agostino)
- Comparação de distribuições antes e após pré-processamento
"""

from __future__ import annotations

import pandas as pd


def plot_distributions(df: pd.DataFrame, outdir: str | None = None) -> None:
    """Gera histogramas com KDE para todas as colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    outdir : str | None, default=None
        Diretório para salvar os gráficos. Se None, exibe na tela.

    Extensão futura
    ---------------
    - Parâmetro ``columns`` para selecionar colunas específicas
    - Colorir por coluna ``target`` quando disponível
    - Adicionar linhas de referência (média, mediana, ±1σ)
    """
    raise NotImplementedError("plot_distributions ainda não foi implementado.")


def test_normality(series: pd.Series) -> dict:
    """Aplica testes de normalidade em uma série numérica.

    Aplica o teste de Shapiro-Wilk (para n ≤ 5000) ou o teste de
    D'Agostino-Pearson para amostras maiores.

    Parâmetros
    ----------
    series : pd.Series
        Série numérica de entrada.

    Retorna
    -------
    dict
        Dicionário com:
        - ``test``      : str — nome do teste aplicado
        - ``statistic`` : float — estatística do teste
        - ``p_value``   : float — p-valor
        - ``is_normal`` : bool — True se p_value > 0.05

    Extensão futura
    ---------------
    - Suporte ao teste de Kolmogorov-Smirnov
    - Suporte a nível de significância configurável
    """
    raise NotImplementedError("test_normality ainda não foi implementado.")

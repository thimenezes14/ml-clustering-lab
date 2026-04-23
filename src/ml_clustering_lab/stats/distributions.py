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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from ml_clustering_lab.utils.io import ensure_dir

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return

    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")

    plt.tight_layout()
    if outdir:
        ensure_dir(outdir)
        fig.savefig(Path(outdir) / "distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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
    from scipy import stats

    clean = series.dropna()
    n = len(clean)

    if n <= 5000:
        stat, p_value = stats.shapiro(clean)
        test_name = "shapiro-wilk"
    else:
        stat, p_value = stats.normaltest(clean)
        test_name = "dagostino-pearson"

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }

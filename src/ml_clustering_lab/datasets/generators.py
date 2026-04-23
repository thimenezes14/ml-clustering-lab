"""
datasets/generators.py
=======================
Módulo gerador de datasets sintéticos para experimentos de clustering.

Responsabilidade
----------------
Fornecer funções para gerar datasets sintéticos com diferentes estruturas,
facilitando o estudo comparativo entre algoritmos de clustering. Cada cenário
foi projetado para destacar pontos fortes e fracos específicos dos algoritmos.

Cenários disponíveis
---------------------
- ``blobs``          : clusters gaussianos separados (ideal para K-Means, Aglomerativo)
- ``moons``          : duas luas entrelaçadas (ideal para DBSCAN e Mean Shift)
- ``circles``        : círculos concêntricos (ideal para DBSCAN)
- ``anisotropic``    : clusters elongados via transformação linear (desafia K-Means)
- ``varied_density`` : clusters com tamanhos e densidades diferentes (desafia DBSCAN)
- ``no_structure``   : pontos uniformemente distribuídos (sem clusters reais)

Comparativo dos cenários
-------------------------
+-------------------+----------+--------+--------------+-----------+
| Cenário           | K-Means  | DBSCAN | Aglomerativo | Mean Shift|
+-------------------+----------+--------+--------------+-----------+
| blobs             | ✅ bom   | ✅ bom | ✅ bom       | ✅ bom    |
| moons             | ❌ ruim  | ✅ bom | ⚠️ ok        | ✅ bom    |
| circles           | ❌ ruim  | ✅ bom | ❌ ruim      | ⚠️ ok     |
| anisotropic       | ⚠️ ok   | ✅ bom | ✅ bom       | ⚠️ ok     |
| varied_density    | ⚠️ ok   | ❌ ruim| ✅ bom       | ⚠️ ok     |
| no_structure      | ❌ ruim  | ✅ bom | ❌ ruim      | ❌ ruim   |
+-------------------+----------+--------+--------------+-----------+

Uso
---
>>> from ml_clustering_lab.datasets.generators import generate
>>> df = generate("blobs", n_samples=300, n_clusters=4)
>>> df.shape
(300, 3)

Extensão futura
---------------
- Cenários com dados de alta dimensionalidade (n_features > 10)
- Gerador de series temporais para clustering temporal
- Datasets com missing values e ruído configurável
- Exportação automática para CSV em ``data/raw/``
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Mapeamento interno: nome → função geradora
_REGISTRY: dict[str, Callable] = {}


def _register(name: str):
    """Decorator para registrar funções geradoras no _REGISTRY."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Funções geradoras individuais
# ---------------------------------------------------------------------------


@_register("blobs")
def generate_blobs(
    n_samples: int = 300,
    n_clusters: int = 3,
    n_features: int = 2,
    cluster_std: float = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera clusters gaussianos bem separados (Gaussian blobs).

    Ideal para verificar o comportamento básico de todos os algoritmos de
    clustering. K-Means e Aglomerativo funcionam bem neste cenário. É o
    ponto de partida clássico para experimentos de clustering.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras.
    n_clusters : int, default=3
        Número de clusters (centroides) a gerar.
    n_features : int, default=2
        Dimensionalidade das amostras.
    cluster_std : float, default=1.0
        Desvio padrão de cada cluster (controla o espalhamento).
    center_box : tuple[float, float], default=(-10.0, 10.0)
        Faixa de valores para posição aleatória dos centroides.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1``, ..., ``xN-1`` e ``label``.

    Exemplo
    -------
    >>> df = generate_blobs(n_samples=200, n_clusters=4)
    >>> df["label"].nunique()
    4

    Extensão futura
    ---------------
    - Aceitar lista de desvios padrões por cluster (``cluster_std`` como lista)
    - Aceitar centros fixos via parâmetro ``centers``
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state,
    )
    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df


@_register("moons")
def generate_moons(
    n_samples: int = 300,
    noise: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera duas luas entrelaçadas (two moons).

    Desafia algoritmos baseados em centroides (K-Means) porque os clusters
    têm forma não convexa. DBSCAN e Mean Shift se saem bem, pois detectam
    a estrutura de densidade local.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras (dividido igualmente entre as duas luas).
    noise : float, default=0.1
        Desvio padrão do ruído gaussiano adicionado. Use 0.0 para dados limpos.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1`` e ``label`` (0 ou 1).

    Exemplo
    -------
    >>> df = generate_moons(n_samples=200, noise=0.05)
    >>> df["label"].unique()
    array([0, 1])

    Extensão futura
    ---------------
    - Parâmetro ``offset`` para separação vertical entre as luas
    """
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["label"] = y
    return df


@_register("circles")
def generate_circles(
    n_samples: int = 300,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera dois círculos concêntricos.

    Extremamente desafiador para K-Means e Aglomerativo com linkage ward.
    DBSCAN detecta a estrutura circular corretamente com os parâmetros certos.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras.
    noise : float, default=0.05
        Ruído gaussiano adicionado. Use 0.0 para círculos perfeitos.
    factor : float, default=0.5
        Razão de escala entre o círculo interno e o externo. Deve estar em (0, 1).
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1`` e ``label`` (0 ou 1).

    Exemplo
    -------
    >>> df = generate_circles(n_samples=200, factor=0.3, noise=0.02)
    >>> df["label"].unique()
    array([0, 1])

    Extensão futura
    ---------------
    - Suporte a múltiplos anéis (mais de 2 círculos concêntricos)
    """
    from sklearn.datasets import make_circles

    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["label"] = y
    return df


@_register("anisotropic")
def generate_anisotropic(
    n_samples: int = 300,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera clusters anisotrópicos (elongados) via transformação linear.

    Os clusters são gerados como blobs gaussianos e em seguida transformados
    com uma matriz de rotação/escala, criando formatos elipsoidais elongados.
    Isso desafia o K-Means, que assume clusters aproximadamente esféricos.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras.
    n_clusters : int, default=3
        Número de clusters a gerar.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1`` e ``label``.

    Exemplo
    -------
    >>> df = generate_anisotropic(n_samples=300, n_clusters=3)
    >>> df.shape
    (300, 3)

    Extensão futura
    ---------------
    - Matriz de transformação configurável pelo usuário
    - Suporte a n_features > 2
    """
    from sklearn.datasets import make_blobs

    rng = np.random.RandomState(random_state)
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)
    # Aplica transformação linear para criar clusters elongados
    transformation = rng.randn(2, 2)
    X = X @ transformation
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["label"] = y
    return df


@_register("varied_density")
def generate_varied_density(
    n_samples: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera clusters com densidades e tamanhos diferentes.

    Combina três clusters com desvios padrões muito diferentes:
    um compacto, um médio e um espalhado. Isso desafia DBSCAN, que assume
    densidade uniforme, e favorece Aglomerativo com linkage ward.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras (distribuído proporcionalmente entre clusters).
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1`` e ``label`` (0, 1 ou 2).

    Exemplo
    -------
    >>> df = generate_varied_density(n_samples=300)
    >>> df["label"].value_counts()
    label
    0    100
    1    100
    2    100
    ...

    Extensão futura
    ---------------
    - Parâmetro ``cluster_stds`` para controlar as variâncias individualmente
    - Suporte a número variável de clusters
    """
    from sklearn.datasets import make_blobs

    # Proporções: cluster compacto, médio, espalhado
    n_each = n_samples // 3
    n_last = n_samples - 2 * n_each

    X, y = make_blobs(
        n_samples=[n_each, n_each, n_last],
        cluster_std=[0.5, 2.0, 4.0],
        centers=[[-5, -5], [0, 5], [5, -5]],
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["label"] = y
    return df


@_register("no_structure")
def generate_no_structure(
    n_samples: int = 300,
    low: float = -10.0,
    high: float = 10.0,
    n_features: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera dados aleatórios sem estrutura de clusters.

    Útil como caso controle: todos os algoritmos devem ter dificuldade com
    este dataset. Silhouette Score próximo de 0 ou negativo indica ausência
    de estrutura real.

    Parâmetros
    ----------
    n_samples : int, default=300
        Número total de amostras.
    low : float, default=-10.0
        Limite inferior da distribuição uniforme.
    high : float, default=10.0
        Limite superior da distribuição uniforme.
    n_features : int, default=2
        Número de dimensões.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1``, ..., ``xN-1``.
        Não há coluna ``label`` (não há clusters verdadeiros).

    Exemplo
    -------
    >>> df = generate_no_structure(n_samples=200)
    >>> "label" in df.columns
    False

    Extensão futura
    ---------------
    - Suporte a distribuições diferentes (normal, exponencial, etc.)
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(low=low, high=high, size=(n_samples, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols)


# ---------------------------------------------------------------------------
# Dispatcher unificado
# ---------------------------------------------------------------------------

#: Nomes de todos os cenários disponíveis.
AVAILABLE_SCENARIOS: list[str] = sorted(_REGISTRY.keys())


def generate(
    kind: str = "blobs",
    n_samples: int = 300,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Gera um dataset sintético pelo nome do cenário.

    Ponto de entrada unificado para todos os geradores do módulo. Delega para
    a função geradora registrada em ``_REGISTRY`` e repassa todos os kwargs.

    Parâmetros
    ----------
    kind : str, default="blobs"
        Nome do cenário. Valores válidos::

            "blobs"          — clusters gaussianos separados
            "moons"          — duas luas entrelaçadas
            "circles"        — círculos concêntricos
            "anisotropic"    — clusters elongados
            "varied_density" — clusters com densidades diferentes
            "no_structure"   — dados sem estrutura de cluster

    n_samples : int, default=300
        Número de amostras a gerar.
    random_state : int, default=42
        Semente para reprodutibilidade.
    **kwargs :
        Parâmetros adicionais específicos de cada gerador.

    Retorna
    -------
    pd.DataFrame
        DataFrame com features e, na maioria dos cenários, coluna ``label``.

    Exceções
    --------
    ValueError
        Se ``kind`` não for um cenário disponível.

    Exemplos
    --------
    >>> df = generate("blobs", n_samples=200, n_clusters=4)
    >>> df.shape
    (200, 3)

    >>> df = generate("moons", n_samples=300, noise=0.05)
    >>> df["label"].nunique()
    2

    >>> df = generate("no_structure", n_samples=100)
    >>> "label" in df.columns
    False

    Ver também
    ----------
    AVAILABLE_SCENARIOS : lista com os nomes de todos os cenários.
    generate_blobs, generate_moons, generate_circles,
    generate_anisotropic, generate_varied_density, generate_no_structure
    """
    key = kind.lower()
    if key not in _REGISTRY:
        available = ", ".join(AVAILABLE_SCENARIOS)
        raise ValueError(
            f"Cenário '{kind}' não encontrado. Disponíveis: {available}"
        )
    return _REGISTRY[key](n_samples=n_samples, random_state=random_state, **kwargs)

"""
datasets/loaders.py
===================
Funções de carregamento de dados de múltiplas fontes.

Responsabilidade
----------------
Fornecer uma interface unificada para carregar datasets independentemente da
origem (arquivo local, URL remota, scikit-learn, datasets sintéticos), sempre
retornando um ``pandas.DataFrame``.

Fontes suportadas (planejadas)
------------------------------
- CSV / Excel / Parquet / JSON do disco local
- URL remota (CSV / JSON)
- Datasets do ``sklearn.datasets`` (iris, wine, breast_cancer, digits, etc.)
- Datasets sintéticos do ``sklearn.datasets`` (make_blobs, make_moons, make_circles)

Extensão futura
---------------
- Suporte a Parquet via ``pandas.read_parquet``
- Suporte a Excel via ``pandas.read_excel``
- Cache local de datasets remotos (evitar re-download)
- Validação de schema após carregamento
- Suporte a autenticação para URLs protegidas
"""

from __future__ import annotations

import pandas as pd


_SKLEARN_LOADERS: dict[str, str] = {
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "digits": "load_digits",
}

_SYNTHETIC_GENERATORS: dict[str, str] = {
    "blobs": "make_blobs",
    "moons": "make_moons",
    "circles": "make_circles",
}


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Carrega um arquivo CSV do disco local.

    Parâmetros
    ----------
    path : str
        Caminho absoluto ou relativo para o arquivo CSV.
    **kwargs :
        Argumentos adicionais repassados para ``pandas.read_csv``
        (ex.: ``sep``, ``encoding``, ``usecols``).

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados.

    Exceções
    --------
    FileNotFoundError
        Se o arquivo não existir no caminho fornecido.
    ValueError
        Se o arquivo não puder ser interpretado como CSV.

    Exemplo
    -------
    >>> df = load_csv("data/raw/iris.csv")

    Extensão futura
    ---------------
    - Adicionar detecção automática de separador via ``pandas.read_csv(sep=None)``
    - Adicionar suporte a arquivos comprimidos (.gz, .bz2)
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o arquivo CSV '{path}': {exc}") from exc


def load_from_url(url: str, **kwargs) -> pd.DataFrame:
    """Baixa e carrega um dataset a partir de uma URL remota.

    Parâmetros
    ----------
    url : str
        URL pública de um arquivo CSV ou JSON.
    **kwargs :
        Argumentos adicionais repassados para ``pandas.read_csv``
        ou ``pandas.read_json``, dependendo da extensão detectada.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados carregados.

    Exceções
    --------
    requests.HTTPError
        Se a URL retornar um status HTTP de erro.
    ValueError
        Se o conteúdo não puder ser interpretado como tabela.

    Exemplo
    -------
    >>> df = load_from_url("https://raw.githubusercontent.com/.../iris.csv")

    Extensão futura
    ---------------
    - Cache local dos arquivos baixados para evitar re-download
    - Suporte a autenticação via header
    - Detecção automática de formato (CSV, JSON, Parquet)
    """
    url_lower = url.lower().split("?")[0]
    try:
        if url_lower.endswith(".json"):
            return pd.read_json(url, **kwargs)
        else:
            return pd.read_csv(url, **kwargs)
    except Exception as exc:
        raise ValueError(f"Não foi possível carregar dados da URL '{url}': {exc}") from exc


def load_sklearn(name: str, as_frame: bool = True) -> pd.DataFrame:
    """Carrega um dataset embutido do scikit-learn.

    Datasets disponíveis
    --------------------
    - ``iris``          : 150 amostras, 4 features, 3 classes
    - ``wine``          : 178 amostras, 13 features, 3 classes
    - ``breast_cancer`` : 569 amostras, 30 features, 2 classes
    - ``digits``        : 1797 amostras, 64 features (pixels), 10 classes

    Parâmetros
    ----------
    name : str
        Nome do dataset (case-insensitive).
    as_frame : bool, default=True
        Se True, retorna um DataFrame com a coluna ``target`` incluída.

    Retorna
    -------
    pd.DataFrame
        DataFrame com features e coluna ``target``.

    Exceções
    --------
    ValueError
        Se ``name`` não corresponder a nenhum dataset disponível.

    Exemplo
    -------
    >>> df = load_sklearn("iris")
    >>> df.shape
    (150, 5)

    Extensão futura
    ---------------
    - Adicionar suporte a datasets adicionais (olivetti_faces, covtype, etc.)
    - Aceitar kwargs extras para customização (ex.: ``return_X_y=True``)
    """
    import sklearn.datasets as skd

    key = name.lower()
    if key not in _SKLEARN_LOADERS:
        available = ", ".join(sorted(_SKLEARN_LOADERS.keys()))
        raise ValueError(f"Dataset sklearn '{name}' não suportado. Disponíveis: {available}")

    loader_fn = getattr(skd, _SKLEARN_LOADERS[key])
    bunch = loader_fn(as_frame=as_frame)

    if as_frame:
        df: pd.DataFrame = bunch.frame.copy()
    else:
        df = pd.DataFrame(bunch.data, columns=[f"x{i}" for i in range(bunch.data.shape[1])])
        df["target"] = bunch.target

    return df


def load_synthetic(
    kind: str = "blobs",
    n_samples: int = 300,
    n_features: int = 2,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Gera um dataset sintético.

    Delega para ``ml_clustering_lab.datasets.generators.generate()``, suportando
    todos os cenários disponíveis em ``AVAILABLE_SCENARIOS``.

    Tipos disponíveis
    -----------------
    - ``blobs``          : clusters gaussianos separados
    - ``moons``          : duas luas entrelaçadas
    - ``circles``        : círculos concêntricos
    - ``anisotropic``    : clusters elongados via transformação linear
    - ``varied_density`` : clusters com densidades diferentes
    - ``no_structure``   : dados aleatórios sem estrutura de clusters

    Parâmetros
    ----------
    kind : str, default="blobs"
        Tipo de dataset sintético.
    n_samples : int, default=300
        Número de amostras a gerar.
    n_features : int, default=2
        Número de features (repassado como kwarg quando suportado pelo gerador).
    random_state : int, default=42
        Semente para reprodutibilidade.
    **kwargs :
        Argumentos adicionais repassados para a função geradora.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1``, ..., ``xN`` e coluna ``label``
        (exceto para ``no_structure``).

    Exceções
    --------
    ValueError
        Se ``kind`` não for um tipo suportado.

    Exemplo
    -------
    >>> df = load_synthetic("moons", n_samples=200)
    >>> df.shape
    (200, 3)

    Extensão futura
    ---------------
    - Suporte a mais tipos de datasets sintéticos
    - Adição de ruído configurável
    - Retorno de metadados (centros dos blobs, etc.)
    """
    from ml_clustering_lab.datasets.generators import AVAILABLE_SCENARIOS, generate

    key = kind.lower()
    if key not in AVAILABLE_SCENARIOS:
        available = ", ".join(sorted(AVAILABLE_SCENARIOS))
        raise ValueError(f"Tipo sintético '{kind}' não suportado. Disponíveis: {available}")

    # Pass n_features only when the generator accepts it (blobs and no_structure do)
    _accepts_n_features = {"blobs", "no_structure"}
    if key in _accepts_n_features:
        kwargs.setdefault("n_features", n_features)

    return generate(kind=key, n_samples=n_samples, random_state=random_state, **kwargs)

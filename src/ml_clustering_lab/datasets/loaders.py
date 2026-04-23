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
    raise NotImplementedError("load_csv ainda não foi implementado.")


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
    raise NotImplementedError("load_from_url ainda não foi implementado.")


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
    raise NotImplementedError("load_sklearn ainda não foi implementado.")


def load_synthetic(
    kind: str = "blobs",
    n_samples: int = 300,
    n_features: int = 2,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Gera um dataset sintético usando scikit-learn.

    Tipos disponíveis
    -----------------
    - ``blobs``   : clusters gaussianos separados (bom para K-Means e Aglomerativo)
    - ``moons``   : duas luas entrelaçadas (bom para DBSCAN e Mean Shift)
    - ``circles`` : círculos concêntricos (bom para DBSCAN)
    - ``aniso``   : clusters anisotrópicos
    - ``varied``  : clusters com variâncias diferentes

    Parâmetros
    ----------
    kind : str, default="blobs"
        Tipo de dataset sintético.
    n_samples : int, default=300
        Número de amostras a gerar.
    n_features : int, default=2
        Número de features (apenas para ``blobs``).
    random_state : int, default=42
        Semente para reprodutibilidade.
    **kwargs :
        Argumentos adicionais repassados para a função geradora do sklearn.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``x0``, ``x1``, ..., ``xN`` e coluna ``label``.

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
    raise NotImplementedError("load_synthetic ainda não foi implementado.")

# Guia de Contribuição

Obrigado por contribuir com o **ml-clustering-lab**! Este guia explica como
adicionar novos algoritmos, datasets e métricas ao projeto, além de descrever
o fluxo de trabalho geral.

---

## Configuração do ambiente de desenvolvimento

```bash
# 1. Clone o repositório
git clone https://github.com/thimenezes14/ml-clustering-lab.git
cd ml-clustering-lab

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Instale o pacote com dependências de desenvolvimento
pip install -e ".[dev]"

# 4. Execute os testes para garantir que tudo está funcionando
python -m pytest tests/ -v
```

---

## Como adicionar um novo algoritmo de clustering

Todos os algoritmos herdam de `ClusteringBase` e devem implementar as propriedades
e métodos obrigatórios da interface.

### Passo a passo

**1. Crie o arquivo do algoritmo em `src/ml_clustering_lab/clustering/`:**

```python
# src/ml_clustering_lab/clustering/meu_algoritmo.py
from __future__ import annotations

import numpy as np
from ml_clustering_lab.clustering.base import ClusteringBase


class MeuAlgoritmoRunner(ClusteringBase):
    """Runner do algoritmo Meu Algoritmo."""

    def __init__(self, param1: int = 5) -> None:
        self.param1 = param1

    @property
    def name(self) -> str:
        return "Meu Algoritmo"

    @property
    def supports_noise(self) -> bool:
        return False  # True se o algoritmo atribui label -1 a ruídos

    @property
    def requires_k(self) -> bool:
        return False  # True se exige n_clusters como parâmetro

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans  # exemplo
        model = MiniBatchKMeans(n_clusters=self.param1)
        return model.fit_predict(X)

    def get_params(self) -> dict:
        return {"param1": self.param1}
```

**2. Registre o algoritmo em `src/ml_clustering_lab/clustering/__init__.py`:**

```python
from ml_clustering_lab.clustering.meu_algoritmo import MeuAlgoritmoRunner

ALGORITHM_REGISTRY: dict[str, type] = {
    ...
    "meu-algoritmo": MeuAlgoritmoRunner,  # nome usado no CLI
}
```

**3. Adicione testes em `tests/test_clustering.py`:**

```python
class TestMeuAlgoritmoRunner:
    def test_instantiation_defaults(self):
        runner = MeuAlgoritmoRunner()
        assert runner.param1 == 5

    def test_fit_predict_shape(self, simple_X):
        runner = MeuAlgoritmoRunner()
        labels = runner.fit_predict(simple_X)
        assert labels.shape == (simple_X.shape[0],)
```

**4. Agora o algoritmo estará disponível no CLI:**

```bash
ml-lab cluster --dataset iris --algorithm meu-algoritmo
```

---

## Como adicionar um novo dataset (cenário sintético)

Os cenários sintéticos são definidos em `src/ml_clustering_lab/datasets/generators.py`
usando o decorator `@_register("nome_do_cenario")`.

### Passo a passo

**1. Adicione a função geradora em `generators.py`:**

```python
@_register("meu_cenario")
def generate_meu_cenario(
    n_samples: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Descrição do cenário."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["label"] = y
    return df
```

**2. Exporte a função em `datasets/__init__.py`:**

```python
from ml_clustering_lab.datasets.generators import (
    ...
    generate_meu_cenario,
)
```

**3. Registre o dataset em `datasets/registry.py` (dentro de `_register_defaults`):**

```python
DatasetInfo(
    name="meu_cenario",
    source="synthetic",
    description="Descrição curta do cenário.",
    loader_kwargs={"kind": "meu_cenario"},
),
```

**4. Adicione testes em `tests/test_loaders.py`.**

---

## Como adicionar uma nova métrica de avaliação

As métricas ficam em `src/ml_clustering_lab/clustering/evaluation.py`.

- **Métricas internas** (não precisam de ground truth): adicione ao retorno de `compute_internal_metrics()`.
- **Métricas externas** (precisam de labels verdadeiros): adicione ao retorno de `compute_external_metrics()`.

---

## Fluxo de trabalho

```
1. Crie um branch: git checkout -b feature/minha-contribuição
2. Implemente a mudança seguindo as convenções do projeto
3. Adicione testes para o novo código
4. Execute os testes: python -m pytest tests/ -v
5. Execute o linter: ruff check src/ tests/
6. Abra um Pull Request descrevendo as mudanças
```

---

## Convenções do projeto

| Aspecto | Convenção |
|---|---|
| Layout do pacote | `src/` layout (PEP 517) |
| Linter/Formatter | `ruff` (configurado em `pyproject.toml`) |
| Testes | `pytest` com classes organizadas por módulo |
| Docstrings | Estilo NumPy com seções: Parâmetros, Retorna, Exceções, Exemplo |
| Type hints | Sempre em funções e métodos públicos |
| Idioma | Código em inglês; docstrings e comentários em português |
| Imports lazy | Imports pesados (`sklearn`, `matplotlib`) dentro das funções |

---

## Estrutura do projeto

```
src/ml_clustering_lab/
├── cli.py                   # CLI (Typer)
├── config.py                # Constantes e paths globais
├── clustering/
│   ├── base.py              # ClusteringBase (interface)
│   ├── kmeans.py            # KMeansRunner
│   ├── dbscan.py            # DBSCANRunner
│   ├── agglomerative.py     # AgglomerativeRunner
│   ├── mean_shift.py        # MeanShiftRunner
│   ├── optimal_k.py         # Método do cotovelo e Silhouette Analysis
│   └── evaluation.py        # compute_internal_metrics, compute_external_metrics
├── datasets/
│   ├── generators.py        # Cenários sintéticos
│   ├── loaders.py           # load_csv, load_sklearn, load_synthetic
│   └── registry.py          # DatasetRegistry
├── pipeline/
│   ├── run_single.py        # Pipeline de experimento único
│   └── run_compare.py       # Pipeline comparativo
├── preprocessing/
│   ├── cleaning.py          # drop_missing, drop_duplicates, remove_outliers
│   ├── feature_selection.py # select_numeric_features
│   └── scaling.py           # scale_features
├── stats/
│   ├── descriptive.py       # describe_dataframe, central_tendency, ...
│   └── distributions.py
├── visualization/
│   ├── plots.py             # plot_histogram, plot_boxplot, ...
│   └── embeddings.py        # reduce_pca, reduce_tsne, plot_pca_2d
└── utils/
    └── io.py                # ensure_dir, save_json, save_dataframe, ...
```

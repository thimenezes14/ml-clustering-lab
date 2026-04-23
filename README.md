# 🔬 ml-clustering-lab

> **Laboratório modular de clustering não supervisionado e análise estatística descritiva em Python.**
> Ideal para praticar Machine Learning, comparar algoritmos e explorar datasets com uma CLI simples e reproduzível.

---

## 📋 Índice

1. [Objetivo](#objetivo)
2. [Visão Geral da Arquitetura](#visão-geral-da-arquitetura)
3. [Estrutura de Diretórios](#estrutura-de-diretórios)
4. [Módulos Principais](#módulos-principais)
5. [Algoritmos Implementados](#algoritmos-implementados)
6. [Instalação](#instalação)
7. [Comandos CLI](#comandos-cli)
8. [Exemplos de Uso](#exemplos-de-uso)
9. [Extensão Futura](#extensão-futura)
10. [Tecnologias](#tecnologias)
11. [Licença](#licença)

---

## Objetivo

`ml-clustering-lab` é um **repositório laboratório** para praticar:

- **Algoritmos de clustering não supervisionado** — K-Means, DBSCAN, Aglomerativo, Mean Shift
- **Análise estatística descritiva** — tendência central, dispersão, distribuição, outliers
- **Carregamento de datasets** — disco local, URL remota e datasets prontos do `scikit-learn`
- **Visualização** — histogramas, boxplots, scatter de clusters, PCA 2D, matrizes de correlação
- **Modo comparativo** — executa todos os algoritmos sobre o mesmo dataset e gera ranking por métricas

O foco é **clareza, modularidade e reprodutibilidade**. Cada etapa é separada em um módulo independente para facilitar o estudo e a extensão futura.

---

## Visão Geral da Arquitetura

```
datasets  →  preprocessing  →  stats  →  clustering  →  evaluation
   ↑                                          ↓               ↓
loaders                                  pipeline         visualization
   ↑                                          ↓               ↓
registry                                  run_single      plots/embeddings
                                          run_compare
                                              ↕
                                           CLI (Typer)
```

O projeto é estruturado como **biblioteca + CLI + notebooks**:

| Camada       | Responsabilidade                                           |
|--------------|-------------------------------------------------------------|
| `datasets`   | Carregamento de dados de múltiplas fontes                  |
| `preprocessing` | Limpeza, normalização, seleção de features             |
| `stats`      | Estatística descritiva e distribuições                     |
| `visualization` | Gráficos exploratórios e de resultados                  |
| `clustering` | Algoritmos de clustering com interface uniforme            |
| `pipeline`   | Orquestração de experimentos (single e comparativo)        |
| `utils`      | IO, logging, helpers gerais                                |
| `cli`        | Interface de linha de comando com Typer                    |

---

## Estrutura de Diretórios

```text
ml-clustering-lab/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/                   # datasets originais (ignorado pelo git)
│   └── processed/             # datasets transformados (ignorado pelo git)
│
├── outputs/
│   ├── figures/               # gráficos gerados (ignorado pelo git)
│   ├── reports/               # relatórios e métricas (ignorado pelo git)
│   └── runs/                  # artefatos de execução (ignorado pelo git)
│
├── notebooks/
│   ├── 01_eda.ipynb           # análise exploratória e estatística
│   ├── 02_clustering.ipynb    # execução individual de algoritmos
│   └── 03_comparison.ipynb    # modo comparativo
│
├── src/
│   └── ml_clustering_lab/
│       ├── __init__.py        # versão e exports públicos
│       ├── cli.py             # CLI Typer com todos os comandos
│       ├── config.py          # configurações e constantes globais
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── loaders.py     # funções de carregamento de dados
│       │   └── registry.py    # registro de datasets disponíveis
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── cleaning.py    # tratamento de nulos, duplicados, outliers
│       │   ├── scaling.py     # normalização e padronização
│       │   └── feature_selection.py  # seleção de features numéricas
│       │
│       ├── stats/
│       │   ├── __init__.py
│       │   ├── descriptive.py # medidas de tendência e dispersão
│       │   └── distributions.py  # análise de distribuições
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── plots.py       # histogramas, boxplots, correlação, scatter
│       │   └── embeddings.py  # PCA, t-SNE, UMAP para visualização 2D
│       │
│       ├── clustering/
│       │   ├── __init__.py
│       │   ├── base.py        # classe base e interface comum
│       │   ├── kmeans.py      # K-Means
│       │   ├── dbscan.py      # DBSCAN
│       │   ├── agglomerative.py  # Clustering Aglomerativo
│       │   ├── mean_shift.py  # Mean Shift
│       │   └── evaluation.py  # métricas de avaliação de clustering
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── run_single.py  # executa um único algoritmo
│       │   └── run_compare.py # executa e compara múltiplos algoritmos
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py          # salvar/carregar artefatos
│           └── logging.py     # configuração de logging
│
└── tests/
    ├── test_loaders.py        # testa carregamento de datasets
    ├── test_stats.py          # testa estatísticas descritivas
    └── test_clustering.py     # testa algoritmos de clustering
```

---

## Módulos Principais

### `datasets/` — Carregamento de Dados

| Função/Classe          | Descrição                                               |
|------------------------|---------------------------------------------------------|
| `load_csv(path)`       | Carrega CSV de disco                                    |
| `load_from_url(url)`   | Baixa e carrega dataset de URL                         |
| `load_sklearn(name)`   | Carrega dataset do scikit-learn (iris, wine, etc.)     |
| `load_synthetic(type)` | Gera dataset sintético (blobs, moons, circles)         |
| `DatasetRegistry`      | Registro de datasets disponíveis com metadados         |

### `stats/` — Análise Estatística Descritiva

| Função/Classe              | Descrição                                          |
|----------------------------|----------------------------------------------------|
| `describe_dataframe(df)`   | Resumo completo: shape, tipos, nulos, duplicados   |
| `central_tendency(series)` | Média, mediana, moda                              |
| `dispersion(series)`       | Variância, desvio padrão, IQR, amplitude, CV      |
| `position_measures(series)`| Quartis, percentis configuráveis                  |
| `shape_measures(series)`   | Assimetria (skewness) e curtose (kurtosis)        |
| `detect_outliers(series)`  | Outliers por IQR e z-score                        |
| `plot_distributions(df)`   | Histogramas e KDE para todas as colunas numéricas |

### `clustering/` — Algoritmos de Clustering

| Módulo            | Algoritmo          | Parâmetros principais               |
|-------------------|--------------------|--------------------------------------|
| `kmeans.py`       | K-Means            | `n_clusters`, `init`, `max_iter`    |
| `dbscan.py`       | DBSCAN             | `eps`, `min_samples`, `metric`      |
| `agglomerative.py`| Aglomerativo       | `n_clusters`, `linkage`, `metric`   |
| `mean_shift.py`   | Mean Shift         | `bandwidth`, `bin_seeding`          |

Todos herdam de `ClusteringBase` e expõem a mesma interface:
```python
algo.fit_predict(X)   # → labels (ndarray)
algo.get_params()     # → dict de hiperparâmetros
algo.name             # → str com nome do algoritmo
algo.supports_noise   # → bool (apenas DBSCAN suporta label -1)
algo.requires_k       # → bool (K-Means e Aglomerativo precisam de k)
```

### `pipeline/` — Orquestração

| Módulo           | Descrição                                                      |
|------------------|----------------------------------------------------------------|
| `run_single.py`  | Carrega dados → pré-processa → roda algoritmo → salva outputs |
| `run_compare.py` | Idem para múltiplos algoritmos → gera tabela comparativa      |

### `visualization/` — Visualização

| Função                       | Descrição                                         |
|------------------------------|---------------------------------------------------|
| `plot_histogram(series)`     | Histograma + KDE                                  |
| `plot_boxplot(df)`           | Boxplot de todas as colunas numéricas             |
| `plot_correlation(df)`       | Heatmap de correlação                             |
| `plot_scatter_clusters(X, labels)` | Scatter 2D colorido por cluster             |
| `plot_dendrogram(model)`     | Dendrograma para clustering aglomerativo          |
| `plot_pca_2d(X, labels)`     | Redução PCA + scatter com clusters                |
| `compare_metrics_plot(df)`   | Gráfico de barras das métricas comparativas       |

### `cli.py` — Comandos Disponíveis

| Comando    | Descrição                                               |
|------------|---------------------------------------------------------|
| `stats`    | Análise estatística descritiva de um dataset           |
| `cluster`  | Executa um algoritmo de clustering                     |
| `compare`  | Compara múltiplos algoritmos no mesmo dataset          |
| `dataset`  | Lista ou carrega datasets disponíveis                  |
| `plot`     | Gera visualizações para um dataset                     |

---

## Algoritmos Implementados

### K-Means

- **Tipo**: Baseado em centroides
- **Requer k?**: ✅ Sim
- **Detecta outliers?**: ❌ Não
- **Complexidade**: O(n · k · t · d) onde t = iterações
- **Ideal para**: clusters aproximadamente esféricos e bem separados
- **Cuidados**: muito sensível à escala; inicializar com `k-means++`; usar o método do cotovelo ou silhouette para escolher k

### DBSCAN

- **Tipo**: Baseado em densidade
- **Requer k?**: ❌ Não
- **Detecta outliers?**: ✅ Sim (label = -1)
- **Complexidade**: O(n log n) com índice espacial
- **Ideal para**: clusters de forma arbitrária, dados com ruído
- **Cuidados**: `eps` e `min_samples` são sensíveis à escala dos dados; dificulta clusters com densidades muito diferentes

### Clustering Aglomerativo (Hierarchical)

- **Tipo**: Hierárquico bottom-up
- **Requer k?**: ✅ Sim (ou corte no dendrograma)
- **Detecta outliers?**: ❌ Não
- **Complexidade**: O(n² log n)
- **Ideal para**: datasets pequenos/médios onde a hierarquia dos clusters é relevante
- **Cuidados**: custo computacional cresce com n; linkage `ward` requer distância euclideana

### Mean Shift

- **Tipo**: Baseado em densidade (estimativa de kernel)
- **Requer k?**: ❌ Não (k é inferido automaticamente)
- **Detecta outliers?**: Parcialmente
- **Complexidade**: O(n²) — lento em datasets grandes
- **Ideal para**: datasets pequenos sem número de clusters pré-definido
- **Cuidados**: `bandwidth` controla a granularidade; alto custo computacional

---

## Instalação

### Pré-requisitos
- Python ≥ 3.10
- pip ou `uv`

### Instalação local (modo editável)

```bash
# 1. Clone o repositório
git clone https://github.com/thimenezes14/ml-clustering-lab.git
cd ml-clustering-lab

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instale o pacote em modo editável
pip install -e ".[dev]"
```

### Verificar instalação

```bash
ml-lab --help
```

---

## Comandos CLI

```
Usage: ml-lab [OPTIONS] COMMAND [ARGS]...

  ml-clustering-lab — laboratório de clustering e análise estatística.

Options:
  --help  Show this message and exit.

Commands:
  cluster   Executa um algoritmo de clustering num dataset.
  compare   Compara múltiplos algoritmos no mesmo dataset.
  dataset   Lista datasets disponíveis ou carrega um novo.
  plot      Gera visualizações para um dataset.
  stats     Exibe análise estatística descritiva de um dataset.
```

---

## Exemplos de Uso

### Análise estatística descritiva

```bash
# Dataset embutido do scikit-learn
ml-lab stats --dataset iris

# Arquivo CSV local
ml-lab stats --source data/raw/meus_dados.csv --output outputs/reports/stats.json
```

### Clustering com um único algoritmo

```bash
# K-Means com k=3 no dataset iris
ml-lab cluster --dataset iris --algorithm kmeans --n-clusters 3

# DBSCAN com parâmetros customizados
ml-lab cluster --dataset wine --algorithm dbscan --eps 0.5 --min-samples 5

# Aglomerativo com linkage ward
ml-lab cluster --source data/raw/dados.csv --algorithm agglomerative --n-clusters 4 --linkage ward

# Mean Shift com bandwidth automático
ml-lab cluster --dataset iris --algorithm mean-shift
```

### Modo comparativo

```bash
# Compara todos os algoritmos no dataset iris
ml-lab compare --dataset iris --algorithms kmeans,dbscan,agglomerative,mean-shift

# Com saída em diretório específico
ml-lab compare --dataset wine --outdir outputs/runs/wine_compare
```

### Visualizações

```bash
# Gera todos os gráficos exploratórios
ml-lab plot --dataset iris --outdir outputs/figures/iris

# Só o heatmap de correlação
ml-lab plot --dataset iris --type correlation
```

### Carregamento de dataset

```bash
# Listar datasets disponíveis
ml-lab dataset --list

# Carregar de URL
ml-lab dataset --from-url "https://example.com/dataset.csv" --name meu_dataset
```

---

## Extensão Futura

O projeto foi projetado para crescer de forma incremental:

| Área                | Extensões planejadas                                        |
|---------------------|--------------------------------------------------------------|
| **Algoritmos**      | OPTICS, GMM, HDBSCAN, Spectral Clustering                  |
| **Redução dim.**    | UMAP, t-SNE, autoencoder                                    |
| **Pré-processamento** | PCA pipeline, encoding de categorias, feature engineering |
| **Configuração**    | Arquivos YAML/TOML por experimento                          |
| **Rastreamento**    | MLflow ou DVC para experiment tracking                      |
| **Relatórios**      | Relatórios automáticos em HTML/Markdown                     |
| **Interface**       | Dashboard web com Streamlit ou Gradio                       |
| **Testes**          | Testes completos com cobertura >80%                         |
| **CI/CD**           | GitHub Actions para lint, testes e release                  |

Para adicionar um novo algoritmo:

1. Crie `src/ml_clustering_lab/clustering/meu_algo.py`
2. Herde de `ClusteringBase` e implemente `fit_predict()`
3. Registre em `clustering/__init__.py`
4. Adicione ao CLI em `cli.py`
5. Escreva testes em `tests/test_clustering.py`

---

## Tecnologias

| Categoria        | Bibliotecas                                              |
|------------------|----------------------------------------------------------|
| Core ML          | `scikit-learn`, `scipy`, `numpy`, `pandas`              |
| Visualização     | `matplotlib`, `seaborn`                                  |
| CLI              | `typer`, `rich`                                          |
| Extras opcionais | `umap-learn`, `plotly`, `yellowbrick`                   |
| Dev/Test         | `pytest`, `pytest-cov`, `ruff`, `mypy`                  |

---

## Licença

MIT — veja [LICENSE](LICENSE) para detalhes.

---

*Projeto para fins educacionais e de portfólio. Contribuições e sugestões são bem-vindas!*

"""
config.py
=========
Configurações globais e constantes do projeto ml-clustering-lab.

Responsabilidade
----------------
Centralizar todos os valores configuráveis do projeto em um único lugar, evitando
"magic numbers" espalhados pelo código. Em versões futuras esta configuração poderá
ser carregada de um arquivo YAML ou variáveis de ambiente.

Extensão futura
---------------
- Suporte a arquivos YAML/TOML por experimento
- Integração com Pydantic Settings para validação automática
- Perfis de configuração (dev, prod, test)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Raiz do repositório (dois níveis acima de src/ml_clustering_lab/)
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
RUNS_DIR: Path = OUTPUTS_DIR / "runs"

NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# ---------------------------------------------------------------------------
# Defaults for clustering algorithms
# ---------------------------------------------------------------------------

DEFAULT_KMEANS_N_CLUSTERS: int = 3
DEFAULT_KMEANS_INIT: str = "k-means++"
DEFAULT_KMEANS_MAX_ITER: int = 300
DEFAULT_KMEANS_RANDOM_STATE: int = 42

DEFAULT_DBSCAN_EPS: float = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES: int = 5

DEFAULT_AGGLOMERATIVE_N_CLUSTERS: int = 3
DEFAULT_AGGLOMERATIVE_LINKAGE: str = "ward"

DEFAULT_MEAN_SHIFT_BANDWIDTH: float | None = None  # None → estimado automaticamente

# ---------------------------------------------------------------------------
# Defaults for stats / preprocessing
# ---------------------------------------------------------------------------

DEFAULT_OUTLIER_Z_THRESHOLD: float = 3.0
DEFAULT_PERCENTILES: list[float] = [0.25, 0.50, 0.75]
DEFAULT_SCALER: str = "standard"  # "standard" | "minmax" | "robust"

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

DEFAULT_FIGURE_DPI: int = 150
DEFAULT_FIGURE_FORMAT: str = "png"
DEFAULT_COLORMAP: str = "tab10"

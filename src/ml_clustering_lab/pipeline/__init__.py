"""
pipeline/__init__.py
====================
Submódulo de orquestração de experimentos de clustering.

Responsabilidade
----------------
Coordenar o fluxo completo de um experimento de clustering:
carregamento → pré-processamento → clustering → avaliação → visualização → armazenamento.

Dois modos de execução
-----------------------
- **Single** (``run_single``): executa um único algoritmo sobre um dataset
- **Compare** (``run_compare``): executa múltiplos algoritmos e gera comparação

Extensão futura
---------------
- Suporte a arquivos de configuração YAML por experimento
- Integração com MLflow ou DVC para experiment tracking
- Execução paralela dos algoritmos no modo compare
"""

from ml_clustering_lab.pipeline.run_compare import run_compare
from ml_clustering_lab.pipeline.run_single import run_single

__all__ = ["run_single", "run_compare"]

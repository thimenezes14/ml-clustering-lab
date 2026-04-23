"""
utils/__init__.py
=================
Submódulo de utilitários gerais do projeto.

Responsabilidade
----------------
Fornecer funções auxiliares reutilizáveis por todos os outros módulos:
IO de artefatos, configuração de logging e helpers gerais.

Extensão futura
---------------
- Helpers para validação de inputs
- Utilitários de timing e profiling
"""

from ml_clustering_lab.utils.io import ensure_dir, load_json, save_json
from ml_clustering_lab.utils.logging import get_logger, setup_logging

__all__ = [
    "ensure_dir",
    "save_json",
    "load_json",
    "setup_logging",
    "get_logger",
]

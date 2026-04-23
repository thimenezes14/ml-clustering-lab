"""
utils/logging.py
================
Configuração centralizada de logging para o projeto.

Responsabilidade
----------------
Fornecer um logger configurado de forma consistente para todos os módulos
do projeto. Usa o módulo padrão ``logging`` do Python com formatação
enriquecida via ``rich.logging.RichHandler`` quando disponível.

Extensão futura
---------------
- Suporte a log em arquivo com rotação automática
- Níveis de log configuráveis via variável de ambiente
- Integração com sistemas de observabilidade (Datadog, OpenTelemetry)
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configura o logging da aplicação.

    Deve ser chamado uma única vez, geralmente no entry point (CLI ou script).
    Usa ``rich.logging.RichHandler`` se disponível, caso contrário cai back
    para o handler padrão do Python.

    Parâmetros
    ----------
    level : str, default="INFO"
        Nível de log: ``"DEBUG"`` | ``"INFO"`` | ``"WARNING"`` | ``"ERROR"``.

    Exemplo
    -------
    >>> setup_logging("DEBUG")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt="[%X]",
        handlers=[handler],
    )


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger com o nome especificado.

    Parâmetros
    ----------
    name : str
        Nome do logger. Convenção: use ``__name__`` no módulo chamador.

    Retorna
    -------
    logging.Logger

    Exemplo
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Iniciando clustering...")
    """
    return logging.getLogger(name)

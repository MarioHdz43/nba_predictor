# -*- coding: utf-8 -*-
"""
infrastructure/logging_config.py
-------------------------------
Configuración centralizada de logging.
Provee loggers configurados según config.yaml.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

from infrastructure.config import config


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Configura el logging centralizado del sistema.
    Debe llamarse una sola vez al inicio.

    Args:
        log_file: Ruta opcional a archivo de log. Si es None, usa solo stderr.
    """
    level_str = config.log_level
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_format = config.log_format
    date_format = config._config.get("logging", {}).get("date_format", "%H:%M:%S")

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(
        logging.Formatter(fmt=log_format, datefmt=date_format)
    )
    handlers.append(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt=log_format, datefmt=date_format)
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    logging.getLogger("nba_api").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger configurado para el módulo dado.

    Args:
        name: Nombre del logger (típicamente __name__ del módulo).

    Returns:
        Logger configurado.
    """
    return logging.getLogger(name)
# -*- coding: utf-8 -*-
"""
domain/prediction_service.py
-----------------------------
Fachada de predicción para compatibilidad hacia atrás.
Delega al PredictionEngine configurado.

Para mejorar la precisión, este módulo puede reemplazarse por un modelo
ML entrenado (XGBoost, red neuronal) sin tocar ninguna otra capa.
"""

from __future__ import annotations
from typing import Optional

from domain.entities import PredictionResult, TeamStats
from domain.prediction_engine import PredictionEngine
from domain.engines.heuristic_engine import HeuristicEngine
from infrastructure.config import config


_engine: Optional[PredictionEngine] = None


def _get_engine() -> PredictionEngine:
    global _engine
    if _engine is None:
        _engine = HeuristicEngine()
    return _engine


def set_engine(engine: PredictionEngine) -> None:
    """Inyecta un motor de predicción diferente."""
    global _engine
    _engine = engine


def normalization_factor(stat: str) -> float:
    """Retorna el factor de normalización de una estadística. API pública."""
    return config.normalization.get(stat, 1.0)


def predict(
    home_stats: TeamStats,
    away_stats: TeamStats,
    home_name: str,
    away_name: str,
) -> PredictionResult:
    """
    Calcula la probabilidad de victoria para cada equipo.
    Delega al motor de predicción configurado.

    Returns:
        PredictionResult sin cuotas. Llamar a .attach_odds(odds)
        para calcular el Expected Value real.
    """
    return _get_engine().predict(home_stats, away_stats, home_name, away_name)

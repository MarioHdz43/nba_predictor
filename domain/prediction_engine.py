# -*- coding: utf-8 -*-
"""
domain/prediction_engine.py
---------------------------
Interfaz abstrata para motores de predicción.
Permite swapped de algoritmos (heuristic, ML, ensemble) sin modificar la aplicación.
"""

from abc import ABC, abstractmethod
from typing import List

from domain.entities import PredictionResult, TeamStats


class PredictionEngine(ABC):
    """
    Contrato para el motor de predicción.
    Implementaciones: HeuristicEngine, XGBoostEngine, EnsembleEngine, etc.
    """

    @abstractmethod
    def predict(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_name: str,
        away_name: str,
    ) -> PredictionResult:
        """
        Calcula la probabilidad de victoria para cada equipo.

        Args:
            home_stats: Estadísticas del equipo local.
            away_stats: Estadísticas del equipo visitante.
            home_name: Nombre completo del equipo local.
            away_name: Nombre completo del equipo visitante.

        Returns:
            PredictionResult sin cuotas. Llamar a .attach_odds(odds)
            para calcular el Expected Value real.
        """
        ...

    @abstractmethod
    def get_engine_name(self) -> str:
        """Retorna el nombre del motor para logging/métricas."""
        ...

    @abstractmethod
    def get_features_used(self) -> List[str]:
        """Retorna la lista de features que usa este motor."""
        ...
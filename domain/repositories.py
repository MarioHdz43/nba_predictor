# -*- coding: utf-8 -*-
"""
domain/repositories.py
-----------------------
Puertos (interfaces abstractas) que definen los contratos entre el dominio
y la infraestructura. La infraestructura implementa estas interfaces;
el dominio nunca conoce los detalles concretos (CSV, SQL, API, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities import H2HStats, Team, PredictionResult


class TeamRepository(ABC):
    """
    Contrato para almacenar y recuperar datos de equipos.
    Puede implementarse sobre CSV, SQLite, Redis, etc.
    """

    @abstractmethod
    def find(self, team_id: int, season: str) -> Optional[Team]:
        """Retorna el equipo si existe en el repositorio, None si no."""
        ...

    @abstractmethod
    def save(self, team: Team) -> None:
        """Persiste (inserta o actualiza) un equipo."""
        ...

    @abstractmethod
    def delete(self, team_id: int, season: str) -> None:
        """Elimina los datos del equipo para esa temporada."""
        ...


class PredictionRepository(ABC):
    """
    Contrato para persistir el historial de predicciones.
    """

    @abstractmethod
    def save(self, prediction: PredictionResult) -> None:
        """Guarda una predicción en el historial."""
        ...

    @abstractmethod
    def list_all(self) -> List[PredictionResult]:
        """Retorna todas las predicciones almacenadas."""
        ...


class TeamDataProvider(ABC):
    """
    Contrato para fuentes de datos externas (NBA API, mock, etc.).
    Separa la obtención de datos en tiempo real de la persistencia.
    """

    @abstractmethod
    def fetch(self, team_id: int, abbreviation: str, season: str) -> Optional[Team]:
        """
        Obtiene datos frescos del equipo desde la fuente externa.
        Retorna None si la fuente no está disponible.
        """
        ...


class H2HRepository(ABC):
    """
    Contrato para obtener historial head-to-head entre equipos.
    """

    @abstractmethod
    def get_h2h(self, home_team: str, away_team: str) -> Optional[H2HStats]:
        """
        Retorna estadísticas H2H. None si no hay historial.
        """
        ...

# -*- coding: utf-8 -*-
"""
application/use_cases.py
-------------------------
Casos de uso de la aplicación. Orquestan el flujo entre el dominio y la
infraestructura usando los puertos definidos en domain/repositories.py.
No contienen lógica de negocio (eso está en el dominio).
No contienen lógica de I/O (eso está en la infraestructura).
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from domain.entities import H2HStats, Odds, PredictionResult, Team
from domain.prediction_engine import PredictionEngine
from domain.repositories import H2HRepository, PredictionRepository, TeamDataProvider, TeamRepository
from infrastructure.config import config as app_config
from infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class GetTeamDataUseCase:
    """
    Obtiene los datos de un equipo con caché inteligente.

    Estrategia:
    1. Verificar si hay datos frescos en el repositorio local.
    2. Si no, obtener de la fuente externa y persistir.
    3. Si la fuente externa falla, usar caché aunque esté desactualizado.
    4. Si no hay nada, lanzar excepción descriptiva.
    """

    def __init__(
        self,
        repository: TeamRepository,
        provider: TeamDataProvider,
        cache_ttl_hours: int = 2,
    ) -> None:
        self._repo = repository
        self._provider = provider
        self._ttl = cache_ttl_hours

    def execute(
        self,
        team_id: int,
        abbreviation: str,
        season: str,
        force_fresh: bool = False,
    ) -> Team:
        cached = self._repo.find(team_id, season)

        if not force_fresh and cached and not cached.is_stale(self._ttl):
            logger.info("Cache hit para %s (%s)", abbreviation, season)
            return cached

        logger.info("Obteniendo datos frescos para %s...", abbreviation)
        fresh = self._provider.fetch(team_id, abbreviation, season)

        if fresh:
            self._repo.save(fresh)
            logger.info("Datos frescos guardados para %s", abbreviation)
            return fresh

        if cached:
            logger.warning(
                "Proveedor no disponible para %s. Usando caché desactualizado.",
                abbreviation,
            )
            return cached

        raise RuntimeError(
            f"No se pudieron obtener datos para '{abbreviation}'. "
            "Verifica tu conexión a internet y que la NBA API esté accesible."
        )


class PredictGameUseCase:
    """
    Orquesta la predicción de un partido:
    1. Obtiene datos de ambos equipos en paralelo.
    2. Delega la predicción al motor configurado.
    3. Si se proveen cuotas, calcula el Expected Value real.
    4. Persiste el resultado en el historial.
    """

    def __init__(
        self,
        get_team_data: GetTeamDataUseCase,
        prediction_repo: PredictionRepository,
        prediction_engine: Optional[PredictionEngine] = None,
        h2h_repo: Optional[H2HRepository] = None,
    ) -> None:
        self._get_team = get_team_data
        self._pred_repo = prediction_repo
        self._engine = prediction_engine
        self._h2h_repo = h2h_repo

    def execute(
        self,
        home_team_id: int,
        home_abbr: str,
        away_team_id: int,
        away_abbr: str,
        season: str,
        force_fresh: bool = False,
        odds: Optional[Odds] = None,
    ) -> PredictionResult:

        # Fetch paralelo de ambos equipos
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    self._get_team.execute,
                    home_team_id, home_abbr, season, force_fresh,
                ): "home",
                executor.submit(
                    self._get_team.execute,
                    away_team_id, away_abbr, season, force_fresh,
                ): "away",
            }
            results = {}
            for future in as_completed(futures):
                role = futures[future]
                results[role] = future.result()  # propaga excepciones del use case

        home_team: Team = results["home"]
        away_team: Team = results["away"]

        h2h_stats = None
        if self._h2h_repo:
            h2h_stats = self._h2h_repo.get_h2h(
                home_team.full_name, away_team.full_name
            )

        from domain import prediction_service as ps
        engine = self._engine or ps
        prediction = engine.predict(
            home_stats=home_team.stats,
            away_stats=away_team.stats,
            home_name=home_team.full_name,
            away_name=away_team.full_name,
            home_team=home_team,
            away_team=away_team,
            h2h_stats=h2h_stats,
        )

        # Si se proveen cuotas, calcular EV real
        if odds:
            prediction = prediction.attach_odds(odds)

        # Persistir en historial
        self._pred_repo.save(prediction)
        logger.info(
            "Predicción guardada: %s vs %s → local %.1f%%",
            home_team.full_name,
            away_team.full_name,
            prediction.home_win_prob * 100,
        )

        return prediction

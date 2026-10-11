# -*- coding: utf-8 -*-
"""
API del Sistema de Predicción NBA.
FastAPI con Logging Estructurado.
"""

from __future__ import annotations
import structlog
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from application.use_cases import GetTeamDataUseCase, PredictGameUseCase
from domain.entities import Odds
from domain.engines.heuristic_engine import HeuristicEngine
from infrastructure.config import config
from infrastructure.database import Database, init_db
from infrastructure.logging_config import setup_logging
from infrastructure.sqlite_repositories import (
    SQLiteH2HRepository,
    SQLitePredictionRepository,
    SQLiteTeamRepository,
)
from infrastructure.nba_api_client import NBAApiTeamDataProvider

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI(
    title="NBA Predictor API",
    description="Sistema de Predicción NBA con Machine Learning",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Equipo local (ID, abreviatura o nombre)")
    away_team: str = Field(..., description="Equipo visitante (ID, abreviatura o nombre)")
    season: str = Field(default="2024-25", description="Temporada NBA")
    odds_home: Optional[float] = Field(default=None, description="Cuota decimal equipo local")
    odds_away: Optional[float] = Field(default=None, description="Cuota decimal equipo visitante")


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    confidence: float
    recommended_bet: str
    home_expected_value: Optional[float] = None
    away_expected_value: Optional[float] = None
    key_factors: list[str]
    predicted_at: str


class AccuracyResponse(BaseModel):
    total: int
    correct: int
    accuracy: float


class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Inicialización al arranque."""
    logger.info("iniciando-nba-predictor-api", version="1.0.0")
    
    init_db()
    logger.info("database-inicializada", path=config.database_path)
    
    db = Database()
    
    team_repo = SQLiteTeamRepository(db)
    prediction_repo = SQLitePredictionRepository(db)
    h2h_repo = SQLiteH2HRepository(db)
    
    nba_provider = NBAApiTeamDataProvider()
    prediction_engine = HeuristicEngine()
    
    get_team_data = GetTeamDataUseCase(
        repository=team_repo,
        provider=nba_provider,
        cache_ttl_hours=config.cache_ttl_hours,
    )
    
    app.state.get_team = get_team_data
    app.state.prediction_repo = prediction_repo
    app.state.h2h_repo = h2h_repo
    
    is_trained = getattr(prediction_engine, "is_trained", lambda: True)()
    logger.info("aplicacion-lista", 
               engine=prediction_engine.get_engine_name(),
               trained=is_trained)


@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check del servicio."""
    db = Database()
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "error"
    
    return HealthResponse(
        status="healthy",
        database=db_status,
        model_loaded=True,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predice el resultado de un partido.
    """
    logger.info("prediccion-solicitada",
              home=request.home_team,
              away=request.away_team,
              season=request.season)
    
    try:
        get_team = app.state.get_team
        prediction_repo = app.state.prediction_repo
        h2h_repo = app.state.h2h_repo
        
        from presentation.cli import resolve_team, load_teams_index
        team_df = load_teams_index()
        
        home_id, home_name, home_abbr = resolve_team(team_df, request.home_team)
        away_id, away_name, away_abbr = resolve_team(team_df, request.away_team)
        
        logger.info("equipos-resueltos",
                  home_id=home_id, away_id=away_id,
                  home_name=home_name, away_name=away_name)
        
        odds = None
        if request.odds_home and request.odds_away:
            odds = Odds(home_decimal=request.odds_home, away_decimal=request.odds_away)
            logger.info("cuotas-recibidas",
                      odds_home=request.odds_home,
                      odds_away=request.odds_away)
        
        predict_game = PredictGameUseCase(
            get_team_data=get_team,
            prediction_repo=prediction_repo,
            prediction_engine=HeuristicEngine(),
            h2h_repo=h2h_repo,
        )
        
        result = predict_game.execute(
            home_team_id=home_id,
            home_abbr=home_abbr,
            away_team_id=away_id,
            away_abbr=away_abbr,
            season=request.season,
            force_fresh=False,
            odds=odds,
        )
        
        logger.info("prediccion-completada",
                  home_team=result.home_team,
                  away_team=result.away_team,
                  home_win_prob=result.home_win_prob,
                  confidence=result.confidence,
                  recommended_bet=result.recommended_bet)
        
        return PredictionResponse(
            home_team=result.home_team,
            away_team=result.away_team,
            home_win_prob=result.home_win_prob,
            away_win_prob=result.away_win_prob,
            confidence=result.confidence,
            recommended_bet=result.recommended_bet,
            home_expected_value=result.home_expected_value,
            away_expected_value=result.away_expected_value,
            key_factors=result.key_factors,
            predicted_at=result.predicted_at.isoformat(),
        )
        
    except Exception as e:
        logger.error("prediccion-fallo", error=str(e), home=request.home_team, away=request.away_team)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/accuracy", response_model=AccuracyResponse)
async def get_accuracy() -> AccuracyResponse:
    """
    Obtiene el accuracy del modelo comparando predicciones con resultados reales.
    """
    logger.info("accuracy-solicitado")
    
    prediction_repo = app.state.prediction_repo
    accuracy = prediction_repo.get_accuracy()
    
    logger.info("accuracy-obtenido", 
              total=accuracy["total"],
              correct=accuracy["correct"],
              accuracy=accuracy["accuracy"])
    
    return AccuracyResponse(
        total=accuracy["total"],
        correct=accuracy["correct"],
        accuracy=accuracy["accuracy"],
    )


@app.post("/result")
async def update_result(
    prediction_id: int = Query(..., description="ID de la predicción"),
    actual_result: str = Query(..., description="home_win o away_win")
) -> JSONResponse:
    """
    Actualiza el resultado real de un partido para tracking de accuracy.
    """
    logger.info("resultado-actualizado",
              prediction_id=prediction_id,
              actual_result=actual_result)
    
    prediction_repo = app.state.prediction_repo
    prediction_repo.update_result(prediction_id, actual_result)
    
    return JSONResponse({"status": "ok", "message": "Result updated"})


@app.get("/predictions/recent")
async def recent_predictions(limit: int = Query(default=10, le=100)):
    """Obtiene las predicciones recientes."""
    prediction_repo = app.state.prediction_repo
    predictions = prediction_repo.get_recent_predictions(limit)
    
    return [
        {
            "id": p.id,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "home_win_prob": p.home_win_prob,
            "confidence": p.confidence,
            "predicted_at": p.predicted_at.isoformat(),
            "actual_result": p.actual_result,
        }
        for p in predictions
    ]


def main():
    """Arranque del servidor."""
    port = int(config._config.get("api", {}).get("port", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
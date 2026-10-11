# -*- coding: utf-8 -*-
"""
infrastructure/sqlite_repositories.py
-----------------------------
Implementaciones SQLite de los repositorios con Logging Estructurado.
"""

import json
from datetime import datetime
from typing import List, Optional

import structlog
from domain.entities import (
    H2HStats,
    Odds,
    PlayerStat,
    PredictionResult,
    Team,
    TeamStats,
)
from domain.repositories import H2HRepository, PredictionRepository, TeamRepository
from infrastructure.database import Database

logger = structlog.get_logger()


class SQLiteTeamRepository(TeamRepository):
    """Repositorio de equipos en SQLite con logging."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or Database()

    def find(self, team_id: int, season: str) -> Optional[Team]:
        logger.debug("team-buscando", team_id=team_id, season=season)
        row = self._db.execute_one(
            "SELECT * FROM teams WHERE team_id = ? AND season = ?",
            (team_id, season)
        )
        if not row:
            logger.debug("team-no-encontrado", team_id=team_id, season=season)
            return None
        logger.debug("team-encontrado", team_id=team_id, season=season)
        return self._row_to_team(row)

    def save(self, team: Team) -> None:
        logger.info("team-guardando", 
                  team_id=team.team_id, 
                  full_name=team.full_name,
                  season=team.season)
        stats_json = json.dumps(team.stats.to_dict())
        top_players_json = json.dumps([
            {"name": p.name, "avg_points": p.avg_points, "avg_minutes": p.avg_minutes}
            for p in team.top_players
        ])
        fetched_at = team.fetched_at.isoformat() if team.fetched_at else None

        self._db.execute(
            """INSERT OR REPLACE INTO teams 
               (team_id, full_name, abbreviation, season, stats_json, top_players_json, fetched_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (team.team_id, team.full_name, team.abbreviation, team.season,
             stats_json, top_players_json, fetched_at, team.source)
        )

    def delete(self, team_id: int, season: str) -> None:
        logger.info("team-eliminando", team_id=team_id, season=season)
        self._db.execute(
            "DELETE FROM teams WHERE team_id = ? AND season = ?",
            (team_id, season)
        )

    def _row_to_team(self, row) -> Team:
        stats = TeamStats()
        try:
            stats = TeamStats.from_dict(json.loads(row["stats_json"]))
        except (json.JSONDecodeError, KeyError):
            pass

        players = []
        try:
            players_data = json.loads(row["top_players_json"] or "[]")
            players = [
                PlayerStat(
                    name=p["name"],
                    avg_points=p["avg_points"],
                    avg_minutes=p["avg_minutes"]
                )
                for p in players_data
            ]
        except (json.JSONDecodeError, KeyError):
            pass

        fetched_at = None
        if row["fetched_at"]:
            try:
                fetched_at = datetime.fromisoformat(row["fetched_at"])
            except ValueError:
                pass

        return Team(
            team_id=row["team_id"],
            full_name=row["full_name"],
            abbreviation=row["abbreviation"],
            season=row["season"],
            stats=stats,
            top_players=players,
            fetched_at=fetched_at,
            source=row["source"]
        )


class SQLitePredictionRepository(PredictionRepository):
    """Repositorio de predicciones en SQLite con logging y accuracy tracking."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or Database()

    def save(self, prediction: PredictionResult) -> None:
        logger.info("prediccion-guardando",
                   home_team=prediction.home_team,
                   away_team=prediction.away_team,
                   home_win_prob=prediction.home_win_prob)
        
        key_factors = " | ".join(prediction.key_factors)
        odds_home = prediction.odds.home_decimal if prediction.odds else None
        odds_away = prediction.odds.away_decimal if prediction.odds else None

        self._db.execute(
            """INSERT INTO predictions 
               (predicted_at, home_team, away_team, home_win_prob, away_win_prob, confidence,
                odds_home, odds_away, home_expected_value, away_expected_value,
                recommended_bet, key_factors, engine_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction.predicted_at.isoformat(),
             prediction.home_team, prediction.away_team,
             prediction.home_win_prob, prediction.away_win_prob, prediction.confidence,
             odds_home, odds_away,
             prediction.home_expected_value, prediction.away_expected_value,
             prediction.recommended_bet, key_factors, "heuristic")
        )
        
        logger.info("prediccion-guardada", 
                   home_team=prediction.home_team,
                   away_team=prediction.away_team)

    def list_all(self) -> List[PredictionResult]:
        logger.debug("predicciones-listando-todas")
        rows = self._db.execute(
            "SELECT * FROM predictions ORDER BY predicted_at DESC"
        )
        return [self._row_to_prediction(r) for r in rows]

    def get_accuracy(self) -> dict:
        """Calcula accuracy comparando predicciones con resultados reales."""
        logger.info("accuracy-calculando")
        rows = self._db.execute(
            """SELECT home_win_prob, actual_result FROM predictions 
               WHERE actual_result IS NOT NULL"""
        )
        
        if not rows:
            logger.info("accuracy-sin-datos", total=0)
            return {"total": 0, "correct": 0, "accuracy": 0.0}

        total = len(rows)
        correct = 0
        for row in rows:
            predicted_home_win = row["home_win_prob"] > 0.5
            actual_home_win = row["actual_result"] == "home_win"
            if predicted_home_win == actual_home_win:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info("accuracy-obtenido", 
                     total=total, 
                     correct=correct, 
                     accuracy=accuracy)
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }

    def update_result(self, prediction_id: int, actual_result: str) -> None:
        logger.info("resultado-actualizando",
                   prediction_id=prediction_id,
                   actual_result=actual_result)
        self._db.execute(
            "UPDATE predictions SET actual_result = ? WHERE id = ?",
            (actual_result, prediction_id)
        )
        logger.info("resultado-actualizado", prediction_id=prediction_id)

    def get_recent_predictions(self, limit: int = 10) -> List[PredictionResult]:
        logger.debug("predicciones-recientes", limit=limit)
        rows = self._db.execute(
            "SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_prediction(r) for r in rows]

    def _row_to_prediction(self, row) -> PredictionResult:
        odds = None
        if row["odds_home"] and row["odds_away"]:
            try:
                odds = Odds(
                    home_decimal=float(row["odds_home"]),
                    away_decimal=float(row["odds_away"])
                )
            except (TypeError, ValueError):
                pass

        row_keys = row.keys() if hasattr(row, "keys") else []
        pred_id = row["id"] if "id" in row_keys else None
        actual_result = row["actual_result"] if "actual_result" in row_keys else None

        return PredictionResult(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_win_prob=float(row["home_win_prob"]),
            away_win_prob=float(row["away_win_prob"]),
            confidence=float(row["confidence"]),
            key_factors=row["key_factors"].split(" | ") if row["key_factors"] else [],
            predicted_at=datetime.fromisoformat(row["predicted_at"]),
            odds=odds,
            home_expected_value=float(row["home_expected_value"]) if row["home_expected_value"] else None,
            away_expected_value=float(row["away_expected_value"]) if row["away_expected_value"] else None,
            recommended_bet=row["recommended_bet"] or "No bet",
            id=pred_id,
            actual_result=actual_result,
        )


class SQLiteH2HRepository(H2HRepository):
    """Repositorio H2H en SQLite con logging."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or Database()

    def get_h2h(self, home_team: str, away_team: str) -> Optional[H2HStats]:
        logger.debug("h2h-buscando", home_team=home_team, away_team=away_team)
        row = self._db.execute_one(
            "SELECT * FROM h2h_history WHERE home_team = ? AND away_team = ?",
            (home_team, away_team)
        )
        if not row:
            logger.debug("h2h-no-encontrado", home_team=home_team, away_team=away_team)
            return None
        logger.debug("h2h-encontrado", home_team=home_team, away_team=away_team)
        return H2HStats(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_wins=row["home_wins"],
            away_wins=row["away_wins"],
            total_games=row["total_games"]
        )

    def save_result(self, home_team: str, away_team: str, actual_result: str) -> None:
        logger.info("h2h-guardando-resultado",
                   home_team=home_team,
                   away_team=away_team,
                   actual_result=actual_result)
        is_home_win = actual_result == "home_win"
        
        existing = self._db.execute_one(
            "SELECT * FROM h2h_history WHERE home_team = ? AND away_team = ?",
            (home_team, away_team)
        )
        
        if existing:
            if is_home_win:
                self._db.execute(
                    "UPDATE h2h_history SET home_wins = home_wins + 1, total_games = total_games + 1 WHERE home_team = ? AND away_team = ?",
                    (home_team, away_team)
                )
            else:
                self._db.execute(
                    "UPDATE h2h_history SET away_wins = away_wins + 1, total_games = total_games + 1 WHERE home_team = ? AND away_team = ?",
                    (home_team, away_team)
                )
        else:
            self._db.execute(
                """INSERT INTO h2h_history (home_team, away_team, home_wins, away_wins, total_games, played_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (home_team, away_team, 1 if is_home_win else 0, 0 if is_home_win else 1, 1, datetime.utcnow().isoformat())
            )
        logger.info("h2h-resultado-guardado", home_team=home_team, away_team=away_team)
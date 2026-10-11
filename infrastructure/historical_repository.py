# -*- coding: utf-8 -*-
"""
infrastructure/historical_repository.py
---------------------------------------
Repositorio SQLite para persistencia y consulta de datos históricos de partidos y boxscores.
Diseñado para inserción transaccional por lotes y descarga incremental (idempotente).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import structlog

from infrastructure.database import Database, get_db

logger = structlog.get_logger()


class SQLiteHistoricalRepository:
    """Repositorio para almacenamiento y consulta de datos históricos de la NBA."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or get_db()

    def get_existing_game_ids(self, seasons: Optional[List[str]] = None) -> Set[str]:
        """
        Retorna el conjunto de IDs de partidos que ya existen en la base de datos.
        Permite filtrar por temporadas para optimizar memoria en ingestas incrementales.
        """
        if seasons:
            placeholders = ",".join("?" for _ in seasons)
            query = f"SELECT game_id FROM historical_games WHERE season IN ({placeholders})"
            rows = self._db.execute(query, tuple(seasons))
        else:
            query = "SELECT game_id FROM historical_games"
            rows = self._db.execute(query)

        existing = {str(row["game_id"]) for row in rows}
        logger.debug("historical-existing-games-checked", count=len(existing))
        return existing

    def save_games_batch(
        self,
        games: List[Dict[str, Any]],
        boxscores: List[Dict[str, Any]],
    ) -> int:
        """
        Guarda partidos y sus boxscores por equipo de forma atómica en una única transacción.
        Utiliza INSERT OR IGNORE para garantizar idempotencia estricta.
        """
        if not games:
            return 0

        game_tuples = [
            (
                str(g["game_id"]),
                str(g["season"]),
                str(g["game_date"]),
                int(g["home_team_id"]),
                int(g["away_team_id"]),
                int(g["home_score"]),
                int(g["away_score"]),
                str(g["season_type"]),
            )
            for g in games
        ]

        boxscore_tuples = [
            (
                str(b["game_id"]),
                int(b["team_id"]),
                int(b["is_home"]),
                int(b["pts"]) if b.get("pts") is not None else None,
                int(b["fgm"]) if b.get("fgm") is not None else None,
                int(b["fga"]) if b.get("fga") is not None else None,
                int(b["fg3m"]) if b.get("fg3m") is not None else None,
                int(b["fg3a"]) if b.get("fg3a") is not None else None,
                int(b["ftm"]) if b.get("ftm") is not None else None,
                int(b["fta"]) if b.get("fta") is not None else None,
                int(b["oreb"]) if b.get("oreb") is not None else None,
                int(b["dreb"]) if b.get("dreb") is not None else None,
                int(b["reb"]) if b.get("reb") is not None else None,
                int(b["ast"]) if b.get("ast") is not None else None,
                int(b["stl"]) if b.get("stl") is not None else None,
                int(b["blk"]) if b.get("blk") is not None else None,
                int(b["tov"]) if b.get("tov") is not None else None,
                int(b["pf"]) if b.get("pf") is not None else None,
                float(b["plus_minus"]) if b.get("plus_minus") is not None else 0.0,
            )
            for b in boxscores
        ]

        with self._db.get_cursor() as cursor:
            cursor.executemany(
                """
                INSERT OR IGNORE INTO historical_games (
                    game_id, season, game_date, home_team_id, away_team_id,
                    home_score, away_score, season_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                game_tuples,
            )

            cursor.executemany(
                """
                INSERT OR IGNORE INTO historical_team_boxscores (
                    game_id, team_id, is_home, pts, fgm, fga, fg3m, fg3a,
                    ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                boxscore_tuples,
            )

        logger.info(
            "historical-batch-guardado",
            games_count=len(games),
            boxscores_count=len(boxscores),
        )
        return len(games)

    def get_games_by_season(self, season: str) -> List[Dict[str, Any]]:
        """Recupera los partidos de una temporada específica."""
        rows = self._db.execute(
            """
            SELECT * FROM historical_games
            WHERE season = ?
            ORDER BY game_date ASC, game_id ASC
            """,
            (season,),
        )
        return [dict(r) for r in rows]

    def get_boxscores_by_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Recupera los boxscores de ambos equipos para un partido determinado."""
        rows = self._db.execute(
            """
            SELECT * FROM historical_team_boxscores
            WHERE game_id = ?
            ORDER BY is_home DESC
            """,
            (game_id,),
        )
        return [dict(r) for r in rows]

    def count_games(self) -> int:
        """Retorna la cantidad total de partidos históricos registrados."""
        row = self._db.execute_one("SELECT COUNT(*) AS cnt FROM historical_games")
        return row["cnt"] if row else 0

    def count_boxscores(self) -> int:
        """Retorna la cantidad total de boxscores históricos registrados."""
        row = self._db.execute_one("SELECT COUNT(*) AS cnt FROM historical_team_boxscores")
        return row["cnt"] if row else 0

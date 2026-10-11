# -*- coding: utf-8 -*-
"""
infrastructure/database.py
--------------------------
Configuración y esquema de SQLite con Logging Estructurado.
"""

import sqlite3
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

import structlog
from infrastructure.config import config

logger = structlog.get_logger()


class Database:
    """Gestor de conexión SQLite con logging."""

    _instance: Optional["Database"] = None
    _connection: Optional[sqlite3.Connection] = None
    _custom_path: Optional[Path] = None

    def __new__(cls, db_path: Optional[Any] = None) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if db_path is not None:
            cls._instance._custom_path = Path(db_path)
            # Si cambia la ruta, reiniciar la conexión
            if cls._instance._connection is not None:
                cls._instance._connection.close()
                cls._instance._connection = None
        return cls._instance

    @property
    def db_path(self) -> Path:
        if self._custom_path is not None:
            return self._custom_path
        path = config._config.get("database", {}).get("path", "nba_predictor.db")
        return Path(path)

    def _init_schema(self) -> None:
        """Crea las tablas si no existen."""
        logger.info("db-inicializando-schema", path=str(self.db_path))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    abbreviation TEXT NOT NULL,
                    season TEXT NOT NULL,
                    stats_json TEXT,
                    top_players_json TEXT,
                    fetched_at TEXT,
                    source TEXT DEFAULT 'unknown',
                    UNIQUE(team_id, season)
                )
            """)
            logger.debug("db-tabla-creada", table="teams")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_at TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_win_prob REAL NOT NULL,
                    away_win_prob REAL NOT NULL,
                    confidence REAL NOT NULL,
                    odds_home REAL,
                    odds_away REAL,
                    home_expected_value REAL,
                    away_expected_value REAL,
                    recommended_bet TEXT,
                    key_factors TEXT,
                    actual_result TEXT,
                    engine_used TEXT DEFAULT 'heuristic'
                )
            """)
            logger.debug("db-tabla-creada", table="predictions")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS h2h_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_wins INTEGER DEFAULT 0,
                    away_wins INTEGER DEFAULT 0,
                    total_games INTEGER DEFAULT 0,
                    played_at TEXT,
                    UNIQUE(home_team, away_team)
                )
            """)
            logger.debug("db-tabla-creada", table="h2h_history")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_games (
                    game_id TEXT PRIMARY KEY,
                    season TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    home_team_id INTEGER NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    home_score INTEGER NOT NULL,
                    away_score INTEGER NOT NULL,
                    season_type TEXT NOT NULL
                )
            """)
            logger.debug("db-tabla-creada", table="historical_games")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_team_boxscores (
                    game_id TEXT NOT NULL,
                    team_id INTEGER NOT NULL,
                    is_home INTEGER NOT NULL,
                    pts INTEGER,
                    fgm INTEGER,
                    fga INTEGER,
                    fg3m INTEGER,
                    fg3a INTEGER,
                    ftm INTEGER,
                    fta INTEGER,
                    oreb INTEGER,
                    dreb INTEGER,
                    reb INTEGER,
                    ast INTEGER,
                    stl INTEGER,
                    blk INTEGER,
                    tov INTEGER,
                    pf INTEGER,
                    plus_minus REAL,
                    PRIMARY KEY (game_id, team_id),
                    FOREIGN KEY (game_id) REFERENCES historical_games(game_id) ON DELETE CASCADE
                )
            """)
            logger.debug("db-tabla-creada", table="historical_team_boxscores")

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_date
                ON predictions(predicted_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_teams
                ON predictions(home_team, away_team)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_games_date
                ON historical_games(game_date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_games_season
                ON historical_games(season)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_games_teams
                ON historical_games(home_team_id, away_team_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_boxscores_team
                ON historical_team_boxscores(team_id)
            """)

            conn.commit()
            logger.info(
                "db-schema-inicializado",
                tables=[
                    "teams",
                    "predictions",
                    "h2h_history",
                    "historical_games",
                    "historical_team_boxscores",
                ],
            )

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            logger.debug("db-conectando", path=str(self.db_path))
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON;")
            logger.info("db-conectado", path=str(self.db_path))
        return self._connection

    @contextmanager
    def get_cursor(self):
        """Context manager para ejecutar queries."""
        conn = self._get_connection()
        try:
            yield conn.cursor()
            conn.commit()
        except Exception as e:
            logger.error("db-query-error", error=str(e))
            conn.rollback()
            raise

    def close(self) -> None:
        if self._connection:
            logger.info("db-cerrando")
            self._connection.close()
            self._connection = None

    def execute(self, query: str, params: tuple = ()):
        logger.debug("db-ejecutando", query=query[:50])
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_one(self, query: str, params: tuple = ()):
        logger.debug("db-ejecutando-uno", query=query[:50])
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()


_db = Database()


def get_db() -> Database:
    return _db


def init_db() -> None:
    """Inicializa la base de datos."""
    _db._init_schema()
# -*- coding: utf-8 -*-
"""
tests/test_historical_ingest.py
--------------------------------
Tests unitarios para el esquema de datos históricos, repositorio SQLite y pipeline de ingesta.
Utiliza mocks para aislar las llamadas de red a nba_api.
"""

from __future__ import annotations
import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from infrastructure.database import Database
from infrastructure.historical_repository import SQLiteHistoricalRepository
from scripts.fetch_historical_data import (
    fetch_season_gamelog_with_retry,
    ingest_historical_data,
    parse_gamelog_dataframe,
)


@pytest.fixture
def temp_db(tmp_path):
    """Crea una base de datos SQLite temporal y limpia para testing."""
    db_file = tmp_path / "test_historical.db"
    # Reiniciar singleton si existía
    Database._instance = None
    db = Database(db_file)
    db._init_schema()
    yield db
    db.close()
    Database._instance = None


@pytest.fixture
def sample_gamelog_df():
    """Genera un DataFrame idéntico a la respuesta de LeagueGameLog con 2 partidos (4 filas)."""
    data = [
        # Partido 1: LAL (Local) vs GSW (Visitante)
        {
            "SEASON_ID": "22023",
            "TEAM_ID": 1610612747,
            "TEAM_ABBREVIATION": "LAL",
            "TEAM_NAME": "Los Angeles Lakers",
            "GAME_ID": "0022300001",
            "GAME_DATE": "2023-10-24",
            "MATCHUP": "LAL vs. GSW",
            "WL": "W",
            "PTS": 115,
            "FGM": 42,
            "FGA": 88,
            "FG3M": 12,
            "FG3A": 32,
            "FTM": 19,
            "FTA": 24,
            "OREB": 9,
            "DREB": 35,
            "REB": 44,
            "AST": 27,
            "STL": 7,
            "BLK": 5,
            "TOV": 13,
            "PF": 18,
            "PLUS_MINUS": 8.0,
        },
        {
            "SEASON_ID": "22023",
            "TEAM_ID": 1610612744,
            "TEAM_ABBREVIATION": "GSW",
            "TEAM_NAME": "Golden State Warriors",
            "GAME_ID": "0022300001",
            "GAME_DATE": "2023-10-24",
            "MATCHUP": "GSW @ LAL",
            "WL": "L",
            "PTS": 107,
            "FGM": 39,
            "FGA": 92,
            "FG3M": 14,
            "FG3A": 41,
            "FTM": 15,
            "FTA": 20,
            "OREB": 11,
            "DREB": 31,
            "REB": 42,
            "AST": 24,
            "STL": 8,
            "BLK": 3,
            "TOV": 15,
            "PF": 21,
            "PLUS_MINUS": -8.0,
        },
        # Partido 2: BOS (Local) vs MIA (Visitante)
        {
            "SEASON_ID": "22023",
            "TEAM_ID": 1610612738,
            "TEAM_ABBREVIATION": "BOS",
            "TEAM_NAME": "Boston Celtics",
            "GAME_ID": "0022300002",
            "GAME_DATE": "2023-10-25",
            "MATCHUP": "BOS vs. MIA",
            "WL": "W",
            "PTS": 119,
            "FGM": 44,
            "FGA": 85,
            "FG3M": 16,
            "FG3A": 38,
            "FTM": 15,
            "FTA": 18,
            "OREB": 8,
            "DREB": 36,
            "REB": 44,
            "AST": 30,
            "STL": 6,
            "BLK": 7,
            "TOV": 11,
            "PF": 16,
            "PLUS_MINUS": 12.0,
        },
        {
            "SEASON_ID": "22023",
            "TEAM_ID": 1610612748,
            "TEAM_ABBREVIATION": "MIA",
            "TEAM_NAME": "Miami Heat",
            "GAME_ID": "0022300002",
            "GAME_DATE": "2023-10-25",
            "MATCHUP": "MIA @ BOS",
            "WL": "L",
            "PTS": 107,
            "FGM": 40,
            "FGA": 89,
            "FG3M": 11,
            "FG3A": 30,
            "FTM": 16,
            "FTA": 21,
            "OREB": 10,
            "DREB": 30,
            "REB": 40,
            "AST": 22,
            "STL": 7,
            "BLK": 4,
            "TOV": 12,
            "PF": 19,
            "PLUS_MINUS": -12.0,
        },
    ]
    return pd.DataFrame(data)


class TestHistoricalDatabaseSchema:
    def test_schema_creates_tables_and_indexes(self, temp_db):
        """Verifica que las tablas históricas e índices se hayan creado correctamente."""
        rows = temp_db.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'index')"
        )
        names = {row["name"] for row in rows}

        assert "historical_games" in names
        assert "historical_team_boxscores" in names
        assert "idx_historical_games_date" in names
        assert "idx_historical_games_season" in names
        assert "idx_historical_games_teams" in names
        assert "idx_historical_boxscores_team" in names

    def test_foreign_key_constraint_enforced(self, temp_db):
        """Verifica que no se permita insertar un boxscore con un game_id que no existe en historical_games."""
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                """
                INSERT INTO historical_team_boxscores (
                    game_id, team_id, is_home, pts
                ) VALUES (?, ?, ?, ?)
                """,
                ("NON_EXISTENT_GAME", 1610612747, 1, 100),
            )


class TestGamelogParsing:
    def test_parse_gamelog_dataframe(self, sample_gamelog_df):
        """Verifica el correcto emparejamiento de equipos locales y visitantes."""
        games, boxscores = parse_gamelog_dataframe(
            sample_gamelog_df, season="2023-24", season_type="Regular Season"
        )

        assert len(games) == 2
        assert len(boxscores) == 4

        # Verificar Juego 1: LAL vs GSW
        game_1 = next(g for g in games if g["game_id"] == "0022300001")
        assert game_1["season"] == "2023-24"
        assert game_1["season_type"] == "Regular Season"
        assert game_1["home_team_id"] == 1610612747  # LAL
        assert game_1["away_team_id"] == 1610612744  # GSW
        assert game_1["home_score"] == 115
        assert game_1["away_score"] == 107
        assert game_1["game_date"] == "2023-10-24"

        # Verificar Boxscores del Juego 1
        b_home = next(b for b in boxscores if b["game_id"] == "0022300001" and b["is_home"] == 1)
        b_away = next(b for b in boxscores if b["game_id"] == "0022300001" and b["is_home"] == 0)

        assert b_home["team_id"] == 1610612747
        assert b_home["pts"] == 115
        assert b_home["ast"] == 27
        assert b_home["plus_minus"] == 8.0

        assert b_away["team_id"] == 1610612744
        assert b_away["pts"] == 107
        assert b_away["ast"] == 24
        assert b_away["plus_minus"] == -8.0

    def test_parse_empty_dataframe(self):
        """Un DataFrame vacío debe retornar listas vacías sin errores."""
        games, boxscores = parse_gamelog_dataframe(pd.DataFrame(), "2023-24", "Regular Season")
        assert games == []
        assert boxscores == []


class TestSQLiteHistoricalRepository:
    def test_save_and_retrieve_games_and_boxscores(self, temp_db, sample_gamelog_df):
        """Verifica la persistencia y lectura a través de SQLiteHistoricalRepository."""
        repo = SQLiteHistoricalRepository(temp_db)
        games, boxscores = parse_gamelog_dataframe(
            sample_gamelog_df, season="2023-24", season_type="Regular Season"
        )

        saved = repo.save_games_batch(games, boxscores)
        assert saved == 2
        assert repo.count_games() == 2
        assert repo.count_boxscores() == 4

        # Consultar por temporada
        season_games = repo.get_games_by_season("2023-24")
        assert len(season_games) == 2

        # Consultar boxscores por partido
        game_boxscores = repo.get_boxscores_by_game("0022300001")
        assert len(game_boxscores) == 2
        assert game_boxscores[0]["is_home"] == 1
        assert game_boxscores[1]["is_home"] == 0

    def test_idempotency_does_not_duplicate(self, temp_db, sample_gamelog_df):
        """Verifica que ejecutar múltiples veces el guardado no duplique registros ni lance IntegrityError."""
        repo = SQLiteHistoricalRepository(temp_db)
        games, boxscores = parse_gamelog_dataframe(
            sample_gamelog_df, season="2023-24", season_type="Regular Season"
        )

        repo.save_games_batch(games, boxscores)
        # Segunda inserción de los mismos datos
        repo.save_games_batch(games, boxscores)

        assert repo.count_games() == 2
        assert repo.count_boxscores() == 4

        existing_ids = repo.get_existing_game_ids(["2023-24"])
        assert existing_ids == {"0022300001", "0022300002"}


class TestHistoricalIngestPipeline:
    @patch("scripts.fetch_historical_data.fetch_season_gamelog_with_retry")
    def test_full_pipeline_ingest_and_incremental(
        self, mock_fetch, temp_db, sample_gamelog_df
    ):
        """Valida que el pipeline completo realice la ingesta y respete descargas incrementales."""
        mock_fetch.return_value = sample_gamelog_df
        repo = SQLiteHistoricalRepository(temp_db)

        # Primera ejecución: debe guardar 2 partidos
        summary = ingest_historical_data(
            seasons=["2023-24"],
            season_types=["Regular Season"],
            repository=repo,
            delay_min=0.0,
            delay_max=0.0,
        )

        assert summary["seasons_processed"] == 1
        assert summary["total_games_found"] == 2
        assert summary["new_games_saved"] == 2
        assert summary["new_boxscores_saved"] == 4
        assert repo.count_games() == 2

        # Segunda ejecución con los mismos datos: debe detectar que ya existen y no insertar nada nuevo
        summary_2 = ingest_historical_data(
            seasons=["2023-24"],
            season_types=["Regular Season"],
            repository=repo,
            delay_min=0.0,
            delay_max=0.0,
        )

        assert summary_2["total_games_found"] == 2
        assert summary_2["new_games_saved"] == 0
        assert summary_2["new_boxscores_saved"] == 0
        assert repo.count_games() == 2

    @patch("scripts.fetch_historical_data.time.sleep")
    @patch("scripts.fetch_historical_data.LeagueGameLog")
    def test_fetch_with_retry_succeeds_after_failures(self, mock_league_log, mock_sleep, sample_gamelog_df):
        """Verifica que la función reintente ante fallos de red y retorne datos al tener éxito."""
        mock_instance = MagicMock()
        mock_instance.get_data_frames.return_value = [sample_gamelog_df]

        # Falla dos veces con Timeout/Error y la tercera tiene éxito
        mock_league_log.side_effect = [
            ConnectionError("Timeout de conexión con stats.nba.com"),
            ConnectionError("Servidor ocupado"),
            mock_instance,
        ]

        df = fetch_season_gamelog_with_retry(
            season="2023-24",
            season_type="Regular Season",
            delay_min=0.0,
            delay_max=0.0,
            max_retries=3,
        )

        assert not df.empty
        assert len(df) == 4
        assert mock_league_log.call_count == 3

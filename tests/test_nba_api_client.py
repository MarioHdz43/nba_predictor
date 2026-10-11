# -*- coding: utf-8 -*-
"""
tests/test_nba_api_client.py
----------------------------
Suite de pruebas unitarias para el motor de marcadores resiliente y la Arquitectura
de Fallback Cascade (ESPN Tier 1 -> NBA CDN Tier 2 -> The Odds API / ScoreboardV2 Tier 3).
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from infrastructure.nba_api_client import (
    DEFAULT_TIMEOUT,
    TEAM_CANONICAL_MAP,
    _normalize_date_to_espn,
    create_resilient_session,
    fetch_scoreboard_results,
    standardize_team_code,
)


class TestTeamCanonicalNormalization:
    def test_standardize_espn_and_historical_variants(self):
        """Verifica la conversión de abreviaturas históricas y variantes de ESPN."""
        assert standardize_team_code("GS") == "GSW"
        assert standardize_team_code("PHO") == "PHX"
        assert standardize_team_code("SA") == "SAS"
        assert standardize_team_code("NY") == "NYK"
        assert standardize_team_code("NO") == "NOP"
        assert standardize_team_code("NJ") == "BKN"
        assert standardize_team_code("WSH") == "WAS"
        assert standardize_team_code("UTAH") == "UTA"
        assert standardize_team_code("CHO") == "CHA"
        assert standardize_team_code("BRK") == "BKN"

    def test_standardize_full_franchise_names(self):
        """Verifica la conversión de nombres completos de las franquicias."""
        assert standardize_team_code("Boston Celtics") == "BOS"
        assert standardize_team_code("Philadelphia 76ers") == "PHI"
        assert standardize_team_code("Los Angeles Lakers") == "LAL"
        assert standardize_team_code("Golden State Warriors") == "GSW"
        assert standardize_team_code("Denver Nuggets") == "DEN"
        assert standardize_team_code("Miami Heat") == "MIA"

    def test_standardize_clean_tricodes_and_edge_cases(self):
        """Verifica el manejo de códigos ya estándar, minúsculas y entradas vacías."""
        assert standardize_team_code("BOS") == "BOS"
        assert standardize_team_code("mia") == "MIA"
        assert standardize_team_code("") == ""
        assert standardize_team_code(None) == ""


class TestResilientSession:
    def test_resilient_session_config(self):
        """Verifica que la sesión monte adaptadores con retry policy y headers modernos."""
        session = create_resilient_session()
        assert "https://" in session.adapters
        assert "http://" in session.adapters
        adapter = session.adapters["https://"]
        assert adapter.max_retries.total == 3
        assert 500 in adapter.max_retries.status_forcelist
        assert 429 in adapter.max_retries.status_forcelist
        assert "Mozilla" in session.headers.get("User-Agent", "")
        assert "application/json" in session.headers.get("Accept", "")


class TestDateNormalization:
    def test_normalize_date_to_espn(self):
        """Verifica conversión de múltiples formatos a YYYYMMDD continuo."""
        assert _normalize_date_to_espn("2026-10-10") == "20261010"
        assert _normalize_date_to_espn("10/10/2026") == "20261010"
        assert _normalize_date_to_espn("20261010") == "20261010"
        assert _normalize_date_to_espn("") == ""


class TestFallbackCascadeScoreboard:
    @patch("infrastructure.nba_api_client.create_resilient_session")
    def test_tier1_espn_success(self, mock_create_session):
        """Tier 1 (ESPN): Extrae correctamente partidos oficiales finalizados e ignora en progreso."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "events": [
                {
                    "competitions": [
                        {
                            "status": {
                                "type": {
                                    "name": "STATUS_FINAL",
                                    "completed": True,
                                }
                            },
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "team": {"abbreviation": "BOS"},
                                    "score": "112",
                                },
                                {
                                    "homeAway": "away",
                                    "team": {"abbreviation": "MIA"},
                                    "score": "98",
                                },
                            ],
                        }
                    ]
                },
                {
                    # Partido no finalizado: debe ignorarse
                    "competitions": [
                        {
                            "status": {
                                "type": {
                                    "name": "STATUS_IN_PROGRESS",
                                    "completed": False,
                                }
                            },
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "team": {"abbreviation": "LAL"},
                                    "score": "50",
                                },
                                {
                                    "homeAway": "away",
                                    "team": {"abbreviation": "GSW"},
                                    "score": "48",
                                },
                            ],
                        }
                    ]
                },
            ]
        }
        mock_session.get.return_value = mock_resp

        results = fetch_scoreboard_results(game_date="2026-10-10")

        assert len(results) == 1
        assert ("BOS", "MIA") in results
        assert results[("BOS", "MIA")] == (112, 98)
        assert ("LAL", "GSW") not in results

    @patch("infrastructure.nba_api_client._fetch_scoreboard_espn")
    @patch("infrastructure.nba_api_client.create_resilient_session")
    def test_tier2_fallback_when_espn_fails(self, mock_create_session, mock_espn):
        """Tier 2 (NBA Live CDN): Si Tier 1 falla, degrada a NBA CDN Edge y extrae scores con gameStatus == 3."""
        mock_espn.return_value = {}  # ESPN no devolvió nada o falló

        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        mock_cdn_resp = MagicMock()
        mock_cdn_resp.status_code = 200
        mock_cdn_resp.json.return_value = {
            "scoreboard": {
                "games": [
                    {
                        "gameStatus": 3,
                        "gameStatusText": "Final",
                        "homeTeam": {"teamTricode": "DEN", "score": 115},
                        "awayTeam": {"teamTricode": "UTA", "score": 105},
                    },
                    {
                        "gameStatus": 1,
                        "gameStatusText": "7:00 pm ET",
                        "homeTeam": {"teamTricode": "CHI", "score": 0},
                        "awayTeam": {"teamTricode": "MIL", "score": 0},
                    },
                ]
            }
        }
        mock_session.get.return_value = mock_cdn_resp

        results = fetch_scoreboard_results(game_date="2026-10-10")

        assert ("DEN", "UTA") in results
        assert results[("DEN", "UTA")] == (115, 105)
        assert ("CHI", "MIL") not in results

    @patch("infrastructure.nba_api_client._fetch_scoreboard_espn")
    @patch("infrastructure.nba_api_client._fetch_scoreboard_nba_cdn")
    @patch("infrastructure.nba_api_client._fetch_scoreboard_nba_api_fallback")
    def test_tier3_fallback_when_tier1_and_tier2_fail(self, mock_sb_fallback, mock_cdn, mock_espn):
        """Tier 3: Si ESPN y NBA CDN fallan, degrada a Tier 3 (ScoreboardV2/Odds API) sin interrumpir."""
        mock_espn.return_value = {}
        mock_cdn.return_value = {}
        mock_sb_fallback.return_value = {("DAL", "PHX"): (108, 100)}

        results = fetch_scoreboard_results(game_date="2026-10-10")

        assert ("DAL", "PHX") in results
        assert results[("DAL", "PHX")] == (108, 100)

    @patch("infrastructure.nba_api_client._fetch_scoreboard_espn")
    @patch("infrastructure.nba_api_client._fetch_scoreboard_nba_cdn")
    @patch("infrastructure.nba_api_client._fetch_scoreboard_odds_api")
    @patch("infrastructure.nba_api_client._fetch_scoreboard_nba_api_fallback")
    def test_all_tiers_fail_returns_empty_safely(self, mock_sb, mock_odds, mock_cdn, mock_espn):
        """Si todos los tiers fallan por caídas de red, retorna {} de forma segura."""
        mock_espn.side_effect = requests.exceptions.ConnectionError("Network down")
        mock_cdn.side_effect = requests.exceptions.Timeout("Timeout")
        mock_odds.return_value = {}
        mock_sb.return_value = {}

        results = fetch_scoreboard_results(game_date="2026-10-10")
        assert results == {}

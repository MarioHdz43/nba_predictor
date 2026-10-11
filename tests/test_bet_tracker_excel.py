# -*- coding: utf-8 -*-
"""
tests/test_bet_tracker_excel.py
-------------------------------
Suite de pruebas unitarias y de integración para:
1. Cliente de cuotas en vivo (The Odds API) con matching canónico y tolerancia a fallos.
2. Maquetación del libro mayor en bloques de 2 filas por encuentro (Pre-partido / Post-partido).
3. Motor de liquidación matemática de pronósticos (ACERTÓ / FALLÓ) y apuestas (P&L en unidades).
4. Persistencia en Excel (.xlsx) y resumen de métricas institucionales.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3

from unittest.mock import MagicMock, PropertyMock, patch
import pytest
import pandas as pd
import openpyxl

from domain.entities import Odds, PredictionResult
from infrastructure.odds.live_odds_client import (
    LiveOddsClient,
    canonicalize_team,
    get_game_odds,
    parse_commence_time,
    list_available_games,
    find_market_match,
    get_games_by_filter,
    group_games_by_date,
)
from presentation.cli import format_game_menu_entry, parse_batch_selection, CLIController
from infrastructure.reporting.bet_tracker_excel import (
    COLUMNS_LEDGER,
    SHEET_BETS,
    SHEET_METRICS,
    calculate_metrics_summary,
    format_date_ddmmyyyy,
    format_local_timestamp,
    get_local_now,
    get_user_timezone,
    is_bet_recommended,
    load_ledger,
    log_bet_prediction,
    reset_ledger,
    save_ledger_workbook,
)
from infrastructure.reporting.settle_bets import (
    auto_reconcile_pending_bets,
    parse_ledger_date,
    settle_pending_bets,
    get_pending_bets,
    settle_bet_manually,
)




@pytest.fixture
def temp_excel_path(tmp_path: Path) -> Path:
    return tmp_path / "test_nba_bet_tracking.xlsx"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_historical.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE historical_games (
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
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            confidence REAL NOT NULL,
            actual_result TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db_file


# ==============================================================================
# 1. Tests del Cliente de Cuotas en Vivo (LiveOddsClient)
# ==============================================================================

class TestLiveOddsClient:
    def test_canonicalize_team_aliases(self):
        """Verifica que diversos formatos y abreviaciones resuelvan a la misma clave NBA."""
        assert canonicalize_team("Boston Celtics") == "BOS"
        assert canonicalize_team("celtics") == "BOS"
        assert canonicalize_team("BOS") == "BOS"
        assert canonicalize_team("LA Clippers") == "LAC"
        assert canonicalize_team("Los Angeles Clippers") == "LAC"
        assert canonicalize_team("Los Angeles Lakers") == "LAL"
        assert canonicalize_team("Miami Heat") == "MIA"
        assert canonicalize_team("Unknown FC") is None

    def test_get_game_odds_no_api_key(self):
        """Si no hay API key configurada, retorna None sin lanzar errores."""
        with patch.object(LiveOddsClient, "api_key", new_callable=PropertyMock, return_value=None):
            client = LiveOddsClient(api_key=None)
            res = client.get_game_odds("Boston Celtics", "Miami Heat")
            assert res is None

    @patch("requests.get")
    def test_get_game_odds_success_with_preferred_bookmaker(self, mock_get):
        """Simula respuesta exitosa de The Odds API con Pinnacle y DraftKings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "event_123",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": 1.60},
                                    {"name": "Miami Heat", "price": 2.40},
                                ]
                            }
                        ]
                    },
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": 1.65},
                                    {"name": "Miami Heat", "price": 2.35},
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="test_key")
        result = client.get_game_odds("Celtics", "Heat")

        assert result is not None
        home_odd, away_odd, bookmaker = result
        # Pinnacle tiene prioridad en PREFERRED_BOOKMAKERS
        assert home_odd == 1.65
        assert away_odd == 2.35
        assert bookmaker == "Pinnacle"

    @patch("requests.get")
    def test_get_game_odds_game_not_found_returns_none(self, mock_get):
        """Si el partido no está programado o no tiene líneas activas, retorna None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="test_key")
        res = client.get_game_odds("Boston Celtics", "Golden State Warriors")
        assert res is None

    @patch("requests.get")
    def test_get_game_odds_api_error_returns_none(self, mock_get):
        """Ante error 401, 429 o excepción de red, retorna None de forma segura."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized key"
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="bad_key")
        res = client.get_game_odds("Boston Celtics", "Miami Heat")
        assert res is None

    @patch("requests.get")
    def test_check_connection_success(self, mock_get):
        """Verifica que check_connection capture cuota restante y estado HTTP 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "498", "x-requests-used": "2"}
        mock_response.json.return_value = [{"id": "g1", "home_team": "Boston Celtics", "away_team": "Miami Heat"}]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="valid_key")
        info = client.check_connection()
        assert info["status_code"] == 200
        assert info["valid_key"] is True
        assert info["requests_remaining"] == "498"
        assert info["requests_used"] == "2"
        assert info["games_count"] == 1

    @patch("requests.get")
    def test_check_connection_empty_list_handled_safely(self, mock_get):
        """Verifica que cuando no hay partidos programados, la lista vacía se maneje limpiamente."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "500", "x-requests-used": "0"}
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="valid_key")
        info = client.check_connection()
        assert info["status_code"] == 200
        assert info["valid_key"] is True
        assert info["requests_remaining"] == "500"
        assert info["games_count"] == 0
        assert client.get_game_odds("Boston Celtics", "Miami Heat") is None

    def test_parse_commence_time_pregame(self):
        """Partidos futuros se marcan como is_pregame=True y [DISPONIBLE - HH:MM PM]."""
        from datetime import timezone
        now_utc = datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc)
        future_iso = "2026-10-10T23:30:00Z"

        dt_utc, dt_loc, loc_str, is_pre, st_str, st_tag = parse_commence_time(
            future_iso,
            target_timezone=timezone(timedelta(hours=-5)),
            reference_now_utc=now_utc,
        )
        assert is_pre is True
        assert "Hoy" in loc_str
        assert "06:30 PM" in loc_str
        assert st_tag == "[DISPONIBLE]"
        assert st_str == "[DISPONIBLE - 06:30 PM]"

    def test_parse_commence_time_closed_game(self):
        """Partidos pasados se marcan como is_pregame=False y [EN JUGADA / CERRADO]."""
        from datetime import timezone
        now_utc = datetime(2026, 10, 10, 23, 0, 0, tzinfo=timezone.utc)
        past_iso = "2026-10-10T20:00:00Z"

        dt_utc, dt_loc, loc_str, is_pre, st_str, st_tag = parse_commence_time(
            past_iso,
            target_timezone=timezone(timedelta(hours=-5)),
            reference_now_utc=now_utc,
        )
        assert is_pre is False
        assert st_tag == "[EN JUGADA / CERRADO]"
        assert st_str == "[EN JUGADA / CERRADO]"

    def test_parse_commence_time_invalid_or_none(self):
        """Valores nulos o inválidos no lanzan excepción y retornan cerrado."""
        _, _, loc_str, is_pre, st_str, _ = parse_commence_time(None)
        assert is_pre is False
        assert st_str == "[EN JUGADA / CERRADO]"

    @patch("requests.get")
    def test_list_available_games_structure_and_sorting(self, mock_get):
        """list_available_games devuelve partidos estructurados ordenados cronológicamente."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "495", "x-requests-used": "5"}
        mock_response.json.return_value = [
            {
                "id": "game_late",
                "home_team": "Los Angeles Lakers",
                "away_team": "Golden State Warriors",
                "commence_time": "2026-10-22T02:00:00Z",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Los Angeles Lakers", "price": 1.75}, {"name": "Golden State Warriors", "price": 2.15}]}]
                    }
                ]
            },
            {
                "id": "game_early",
                "home_team": "Detroit Pistons",
                "away_team": "Boston Celtics",
                "commence_time": "2026-10-20T19:00:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Detroit Pistons", "price": 1.83}, {"name": "Boston Celtics", "price": 2.07}]}]
                    }
                ]
            }
        ]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="valid_key")
        games = client.list_available_games()

        assert len(games) == 2
        # Ordenado cronológicamente: Pistons (20 oct) antes que Lakers (22 oct)
        assert games[0]["home_team"] == "Detroit Pistons"
        assert games[0]["home_odds"] == 1.83
        assert games[0]["away_odds"] == 2.07
        assert games[0]["bookmaker"] == "DraftKings"
        assert "local_time_str" in games[0]
        assert "is_pregame" in games[0]

        assert games[1]["home_team"] == "Los Angeles Lakers"
        assert games[1]["bookmaker"] == "Pinnacle"

    @patch("requests.get")
    def test_find_market_match_direct_and_inverted_detection(self, mock_get):
        """find_market_match identifica correctamente coincidencia directa e invertida."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = [
            {
                "id": "g_det_bos",
                "home_team": "Detroit Pistons",
                "away_team": "Boston Celtics",
                "commence_time": "2026-10-20T19:00:00Z",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Detroit Pistons", "price": 1.83}, {"name": "Boston Celtics", "price": 2.07}]}]
                    }
                ]
            }
        ]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="test_key")

        # 1. Consulta directa: Detroit local, Boston visitante
        match_direct = client.find_market_match("Detroit Pistons", "Boston Celtics")
        assert match_direct is not None
        assert match_direct["is_inverted"] is False
        assert match_direct["official_home"] == "Detroit Pistons"
        assert match_direct["official_away"] == "Boston Celtics"
        assert match_direct["home_odds"] == 1.83
        assert match_direct["away_odds"] == 2.07

        # 2. Consulta invertida: Usuario ingresó Boston local y Detroit visitante
        match_inv = client.find_market_match("Boston Celtics", "Detroit Pistons")
        assert match_inv is not None
        assert match_inv["is_inverted"] is True
        assert match_inv["official_home"] == "Detroit Pistons"
        assert match_inv["official_away"] == "Boston Celtics"
        assert match_inv["home_odds"] == 1.83
        assert match_inv["away_odds"] == 2.07

    def test_format_game_menu_entry(self):
        """Verifica que format_game_menu_entry genere el texto esperado del menú."""
        sample_game = {
            "home_team": "Detroit Pistons",
            "away_team": "Boston Celtics",
            "home_odds": 1.83,
            "away_odds": 2.07,
            "local_time_str": "Hoy 06:30 PM",
            "is_pregame": True,
            "status_tag": "[DISPONIBLE]",
        }
        entry = format_game_menu_entry(1, sample_game)
        assert "[1] Detroit Pistons vs Boston Celtics" in entry
        assert "Hora: Hoy 06:30 PM" in entry
        assert "1.83 (Pistons) vs 2.07 (Celtics)" in entry
        assert "[DISPONIBLE]" in entry


# ==============================================================================
# 2. Tests de Registro Bi-Fila en Excel (Esquema de 7 Columnas)
# ==============================================================================

class TestBiRowLedgerLogging:
    def test_log_game_without_odds_is_recorded(self, temp_excel_path: Path):
        """Todo partido analizado se registra en 1 fila, aún sin cuotas (Decisión: NO APOSTAR)."""
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.72,
            away_win_prob=0.28,
            confidence=0.85,
            key_factors=["Margen proyectado: +7.5 pts"],
            projected_margin=7.5,
        )

        logged = log_bet_prediction(
            pred,
            odds=None,
            game_id="0022300001",
            game_date="2026-10-10",
            excel_path=temp_excel_path,
        )
        assert logged is True
        assert temp_excel_path.exists()

        df = load_ledger(temp_excel_path)
        assert len(df) == 1

        # Fila única del encuentro (Pronóstico)
        row1 = df.iloc[0]
        assert row1["Fecha"] == "10/10/2026"
        assert row1["Partido"] == "BOS - MIA"
        assert row1["Cuota"] == "NO DISPONIBLE"
        assert "BOS" in str(row1["Probabilidad"])
        assert "72.0%" in str(row1["Probabilidad"])
        assert row1["Decision"] == "NO APOSTAR"
        assert row1["Marcador_Final"] == "-"
        assert row1["Estado"] == "PENDIENTE"

    def test_log_game_with_value_bet_decision_apostar(self, temp_excel_path: Path):
        """Partido con Edge positivo se registra con Decisión 'APOSTAR' y cuota real."""
        pred = PredictionResult(
            home_team="Denver Nuggets",
            away_team="Utah Jazz",
            home_win_prob=0.68,
            away_win_prob=0.32,
            confidence=0.80,
            key_factors=["Margen proyectado: +6.0 pts"],
            projected_margin=6.0,
        )
        odds = Odds(home_decimal=1.95, away_decimal=1.95)

        logged = log_bet_prediction(
            pred,
            odds=odds,
            game_id="0022300002",
            game_date="2026-10-11",
            excel_path=temp_excel_path,
        )
        assert logged is True

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "11/10/2026"
        assert row1["Partido"] == "DEN - UTA"
        assert row1["Decision"] == "APOSTAR"
        assert "1.95" in str(row1["Cuota"])
        assert "DEN" in str(row1["Probabilidad"])
        assert "68.0%" in str(row1["Probabilidad"])
        assert row1["Marcador_Final"] == "-"
        assert row1["Estado"] == "PENDIENTE"



# ==============================================================================
# 3. Tests de Liquidación de Resultados en 7 Columnas
# ==============================================================================

class TestBiRowSettlement:
    def test_settle_game_won_with_apostar(self, temp_excel_path: Path):
        """
        Pronóstico acertado con recomendación APOSTAR:
        Marcador = 112 - 104 (+8). Fila única pasa a 'ACERTÓ' y recibe el marcador final.
        """
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.65,
            away_win_prob=0.35,
            confidence=0.75,
            key_factors=["BOS superiority"],
            projected_margin=5.5,
        )
        odds = Odds(home_decimal=2.00, away_decimal=1.80)
        log_bet_prediction(pred, odds=odds, game_id="GAME_W1", game_date="2026-10-12", excel_path=temp_excel_path)

        fallback = [{
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "home_score": 112,
            "away_score": 104,
        }]

        result = settle_pending_bets(excel_path=temp_excel_path, fallback_results=fallback)
        assert result["settled"] == 1
        assert result["won"] == 1
        assert result["total_pnl"] > 0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "12/10/2026"
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "112 - 104"

    def test_settle_game_lost_with_apostar(self, temp_excel_path: Path):
        """
        Pronóstico fallado con recomendación APOSTAR:
        Marcador = 98 - 105 (-7). Fila única pasa a 'FALLÓ' y recibe el marcador final.
        """
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.65,
            away_win_prob=0.35,
            confidence=0.75,
            key_factors=["BOS superiority"],
            projected_margin=5.5,
        )
        odds = Odds(home_decimal=2.00, away_decimal=1.80)
        log_bet_prediction(pred, odds=odds, game_id="GAME_L1", game_date="2026-10-12", excel_path=temp_excel_path)

        fallback = [{
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "home_score": 98,
            "away_score": 105,
        }]

        result = settle_pending_bets(excel_path=temp_excel_path, fallback_results=fallback)
        assert result["settled"] == 1
        assert result["lost"] == 1
        assert result["total_pnl"] < 0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "12/10/2026"
        assert row1["Estado"] == "FALLÓ"
        assert row1["Marcador_Final"] == "98 - 105"

    def test_settle_no_apostar_game_updates_accuracy_with_zero_pnl(self, temp_excel_path: Path):
        """
        Partido con 'NO APOSTAR':
        El modelo proyecta BOS y acierta el ganador, marcador se asienta y P&L es 0.0.
        """
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.75,
            key_factors=["No odds match"],
        )
        # Sin cuotas -> Decision = NO APOSTAR
        log_bet_prediction(pred, odds=None, game_id="GAME_NO_BET", game_date="2026-10-12", excel_path=temp_excel_path)

        fallback = [{
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "home_score": 120,
            "away_score": 110,
        }]

        result = settle_pending_bets(excel_path=temp_excel_path, fallback_results=fallback)
        assert result["settled"] == 1
        assert result["total_pnl"] == 0.0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "12/10/2026"
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "120 - 110"

    def test_settle_push_game(self, temp_excel_path: Path):
        """Marcador empatado resulta en 'PUSH' y P&L = 0.0."""
        pred = PredictionResult(
            home_team="Chicago Bulls",
            away_team="Detroit Pistons",
            home_win_prob=0.60,
            away_win_prob=0.40,
            confidence=0.65,
            key_factors=["Even game"],
        )
        odds = Odds(home_decimal=1.90, away_decimal=1.90)
        log_bet_prediction(pred, odds=odds, game_id="GAME_PUSH", game_date="2026-10-12", excel_path=temp_excel_path)

        fallback = [{
            "home_team": "Chicago Bulls",
            "away_team": "Detroit Pistons",
            "home_score": 100,
            "away_score": 100,
        }]

        result = settle_pending_bets(excel_path=temp_excel_path, fallback_results=fallback)
        assert result["settled"] == 1
        assert result["push"] == 1
        assert result["total_pnl"] == 0.0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "12/10/2026"
        assert row1["Estado"] == "PUSH"
        assert row1["Marcador_Final"] == "100 - 100"


# ==============================================================================
# 4. Tests de Persistencia, Estilos y Dashboard en Excel (.xlsx)
# ==============================================================================

class TestExcelPersistenceAndStructure:
    def test_openpyxl_sheets_and_formatting(self, temp_excel_path: Path):
        """Verifica que el archivo contenga las hojas Registro_Apuestas y Resumen_Metricas con estilos."""
        pred = PredictionResult(
            home_team="Golden State Warriors",
            away_team="Los Angeles Lakers",
            home_win_prob=0.62,
            away_win_prob=0.38,
            confidence=0.75,
            key_factors=["GSW edge"],
        )
        log_bet_prediction(pred, odds=None, game_id="GSW_LAL", game_date="2026-10-10", excel_path=temp_excel_path)

        wb = openpyxl.load_workbook(temp_excel_path)
        assert SHEET_BETS in wb.sheetnames
        assert SHEET_METRICS in wb.sheetnames

        ws_bets = wb[SHEET_BETS]
        # Cabecera coloreada y 7 columnas oficiales
        assert ws_bets.max_column == 7
        assert ws_bets.cell(row=1, column=1).fill.start_color.rgb is not None
        assert ws_bets.max_row == 2
        wb.close()

    def test_metrics_summary_aggregate_calculation(self):
        """Verifica el cálculo de KPIs agregados (Accuracy del modelo, Win Rate y Yield de apuestas)."""
        data = [
            # Juego 1: APOSTAR y ACERTÓ (+1.0 u ganada con cuota 2.00)
            {"Fecha": "01/10/2026", "Partido": "BOS - MIA", "Cuota": "2.00", "Probabilidad": "70.0% (BOS)", "Decision": "APOSTAR", "Marcador_Final": "110 - 102", "Estado": "ACERTÓ"},
            # Juego 2: APOSTAR y FALLÓ (-1.0 u)
            {"Fecha": "02/10/2026", "Partido": "LAL - GSW", "Cuota": "1.90", "Probabilidad": "60.0% (LAL)", "Decision": "APOSTAR", "Marcador_Final": "95 - 105", "Estado": "FALLÓ"},
            # Juego 3: NO APOSTAR y ACERTÓ (0.0 u apostadas)
            {"Fecha": "03/10/2026", "Partido": "DEN - UTA", "Cuota": "NO DISPONIBLE", "Probabilidad": "75.0% (DEN)", "Decision": "NO APOSTAR", "Marcador_Final": "108 - 98", "Estado": "ACERTÓ"},
        ]
        df = pd.DataFrame(data)
        metrics = calculate_metrics_summary(df)

        metrics_dict = dict(zip(metrics["Métrica"], metrics["Valor"]))
        assert metrics_dict["Total Encuentros Analizados"] == 3
        assert metrics_dict["Encuentros Liquidados"] == 3
        assert metrics_dict["Pronósticos Acertados"] == 2
        assert metrics_dict["Pronósticos Fallados"] == 1
        assert metrics_dict["Tasa de Acierto del Modelo (Accuracy %)"] == "66.67%"
        assert metrics_dict["Recomendaciones 'APOSTAR'"] == 2
        assert metrics_dict["Recomendaciones 'NO APOSTAR'"] == 1
        assert metrics_dict["Apuestas Ganadas"] == 1
        assert metrics_dict["Apuestas Perdidas"] == 1
        assert metrics_dict["Total Unidades Apostadas"] == 2.0
        assert metrics_dict["P&L Neto Estimado (Unidades)"] == 0.0
        assert "Última Actualización" in metrics_dict
        assert re.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$", str(metrics_dict["Última Actualización"]))

    def test_reset_ledger(self, temp_excel_path: Path):
        """Verifica que reset_ledger genere un Excel limpio con las 7 columnas oficiales."""
        res_path = reset_ledger(temp_excel_path)
        assert res_path.exists()
        df = load_ledger(temp_excel_path)
        assert list(df.columns) == COLUMNS_LEDGER
        assert len(df.columns) == 7
        assert len(df) == 0


# ==============================================================================
# 5. Tests de Integración Cruzada con SQLite
# ==============================================================================

class TestDatabaseCrossReference:
    def test_settle_with_sqlite_historical_games(self, temp_excel_path: Path, temp_db_path: Path):
        """Cruza encuentros pendientes contra la tabla historical_games en SQLite."""
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO historical_games (
                game_id, season, game_date, home_team_id, away_team_id,
                home_score, away_score, season_type
            ) VALUES ('0022300999', '2023-24', '2024-03-01', 1610612738, 1610612747, 125, 110, 'Regular Season')
        """)
        conn.commit()
        conn.close()

        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_win_prob=0.75,
            away_win_prob=0.25,
            confidence=0.85,
            key_factors=["BOS at home"],
            projected_margin=8.0,
        )
        odds = Odds(home_decimal=1.80, away_decimal=2.10)
        log_bet_prediction(pred, odds=odds, game_id="0022300999", game_date="2024-03-01", excel_path=temp_excel_path)

        result = settle_pending_bets(db_path=temp_db_path, excel_path=temp_excel_path)
        assert result["settled"] == 1
        assert result["won"] == 1

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Fecha"] == "01/03/2024"
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "125 - 110"



# ==============================================================================
# 6. Tests de Auto-Liquidación Silenciosa al Iniciar
# ==============================================================================

class TestAutoReconcile:
    def test_auto_reconcile_no_pending_exits_fast(self, temp_excel_path: Path):
        """Si no hay partidos pendientes, finaliza inmediatamente sin demoras ni llamadas."""
        reset_ledger(temp_excel_path)
        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 0

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_with_pending_and_mocked_scoreboard(self, mock_fetch, temp_excel_path: Path):
        """Si hay partidos pendientes finalizados, se liquidan automáticamente al iniciar."""
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.80,
            key_factors=["Superioridad local"],
            projected_margin=7.0,
        )
        odds = Odds(home_decimal=1.85, away_decimal=2.00)
        log_bet_prediction(pred, odds=odds, excel_path=temp_excel_path)

        # Simular retorno de la API para BOS vs MIA
        mock_fetch.return_value = {("BOS", "MIA"): (115, 105)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 1

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "115 - 105"

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_today_game_settles_immediately(self, mock_fetch, temp_excel_path: Path):
        """Partidos programados para HOY (game_date == today) se liquidan de inmediato si finalizaron."""
        today_str = get_local_now().strftime("%Y-%m-%d")
        pred = PredictionResult(
            home_team="Milwaukee Bucks",
            away_team="Chicago Bulls",
            home_win_prob=0.80,
            away_win_prob=0.20,
            confidence=0.85,
            key_factors=["Bucks dominance"],
            projected_margin=9.0,
        )
        odds = Odds(home_decimal=1.40, away_decimal=3.10)
        log_bet_prediction(pred, odds=odds, game_date=today_str, excel_path=temp_excel_path)

        # Simular que el partido de hoy ya terminó oficialmente (Status 3 / Final)
        mock_fetch.return_value = {("MIL", "CHI"): (122, 108)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 1

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "122 - 108"

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_bidirectional_cruce_invertido(self, mock_fetch, temp_excel_path: Path):
        """
        Cruce bidireccional:
        En el Excel se registró 'PHI - BOS' proyectando a BOS como ganador (81.4% BOS).
        En la API de la NBA el local oficial fue BOS y el visitante PHI: BOS 114 - PHI 106.
        El cruce bidireccional asigna PHI=106 y BOS=114 (Marcador: '106 - 114') y valida ACERTÓ.
        """
        pred = PredictionResult(
            home_team="Philadelphia 76ers",
            away_team="Boston Celtics",
            home_win_prob=0.186,
            away_win_prob=0.814,
            confidence=0.80,
            key_factors=["BOS superiority"],
            projected_margin=-8.0,
        )
        odds = Odds(home_decimal=3.50, away_decimal=1.35)
        today_str = get_local_now().strftime("%Y-%m-%d")
        log_bet_prediction(pred, odds=odds, game_date=today_str, excel_path=temp_excel_path)

        # La API retorna con BOS local y PHI visitante
        mock_fetch.return_value = {("BOS", "PHI"): (114, 106)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 1

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Partido"] == "PHI - BOS"
        assert row1["Estado"] == "ACERTÓ"
        assert row1["Marcador_Final"] == "106 - 114"

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_bidirectional_cruce_invertido_fallo(self, mock_fetch, temp_excel_path: Path):
        """
        Cruce bidireccional en fallo:
        En Excel se registró 'PHI - BOS' proyectando a PHI (65% PHI).
        En la API BOS ganó 114 a 106. Marcador '106 - 114'.
        El estado pasa a 'FALLÓ'.
        """
        pred = PredictionResult(
            home_team="Philadelphia 76ers",
            away_team="Boston Celtics",
            home_win_prob=0.65,
            away_win_prob=0.35,
            confidence=0.75,
            key_factors=["PHI edge"],
            projected_margin=4.0,
        )
        odds = Odds(home_decimal=1.80, away_decimal=2.10)
        today_str = get_local_now().strftime("%Y-%m-%d")
        log_bet_prediction(pred, odds=odds, game_date=today_str, excel_path=temp_excel_path)

        mock_fetch.return_value = {("BOS", "PHI"): (114, 106)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 1

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Estado"] == "FALLÓ"
        assert row1["Marcador_Final"] == "106 - 114"

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_future_game_stays_pending(self, mock_fetch, temp_excel_path: Path):
        """Partidos futuros (game_date > local_today) no se liquidan prematuramente."""
        tomorrow_str = (get_local_now() + timedelta(days=1)).strftime("%Y-%m-%d")
        pred = PredictionResult(
            home_team="Denver Nuggets",
            away_team="Utah Jazz",
            home_win_prob=0.85,
            away_win_prob=0.15,
            confidence=0.80,
            key_factors=["DEN altitude"],
        )
        log_bet_prediction(pred, odds=None, game_date=tomorrow_str, excel_path=temp_excel_path)

        # Mock retorna resultado, pero el partido es futuro
        mock_fetch.return_value = {("DEN", "UTA"): (110, 95)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        row1 = df.iloc[0]
        assert row1["Estado"] == "PENDIENTE"
        assert row1["Marcador_Final"] == "-"

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_transparency_console_logs(self, mock_fetch, temp_excel_path: Path, capsys):
        """Verifica la emisión de logs informativos requeridos en consola."""
        today_str = get_local_now().strftime("%Y-%m-%d")
        pred = PredictionResult(
            home_team="Dallas Mavericks",
            away_team="Phoenix Suns",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.75,
            key_factors=["DAL at home"],
        )
        odds = Odds(home_decimal=1.75, away_decimal=2.15)
        log_bet_prediction(pred, odds=odds, game_date=today_str, excel_path=temp_excel_path)

        mock_fetch.return_value = {("DAL", "PHX"): (110, 102)}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 1

        captured = capsys.readouterr()
        assert "[Auto-Liquidación] Verificando 1 partido(s) pendiente(s)..." in captured.out
        assert "[Auto-Liquidación] Consultando marcadores vía ESPN/NBA Network para fecha" in captured.out
        assert "[Auto-Liquidación] Partido 'DAL - PHX' finalizado (Marcador: 110 - 102) -> Liquidado como ACERTÓ." in captured.out

    @patch("infrastructure.nba_api_client.fetch_scoreboard_results")
    def test_auto_reconcile_unsettled_remaining_log(self, mock_fetch, temp_excel_path: Path, capsys):
        """Verifica log informativo cuando partidos pendientes no cuentan con marcador final."""
        today_str = get_local_now().strftime("%Y-%m-%d")
        pred = PredictionResult(
            home_team="Toronto Raptors",
            away_team="Brooklyn Nets",
            home_win_prob=0.55,
            away_win_prob=0.45,
            confidence=0.60,
            key_factors=["Close game"],
        )
        log_bet_prediction(pred, odds=None, game_date=today_str, excel_path=temp_excel_path)

        # API no tiene resultados aún (partido en juego o no iniciado)
        mock_fetch.return_value = {}

        count = auto_reconcile_pending_bets(excel_path=temp_excel_path)
        assert count == 0

        captured = capsys.readouterr()
        assert "[Auto-Liquidación] Verificando 1 partido(s) pendiente(s)..." in captured.out
        assert "[Auto-Liquidación] 1 partido(s) aún no cuentan con marcador final oficial en NBA API." in captured.out

    def test_parse_ledger_date_utility(self):
        """Verifica la función auxiliar parse_ledger_date en diferentes formatos."""
        assert parse_ledger_date("10/10/2026") == date(2026, 10, 10)
        assert parse_ledger_date("2026-10-10") == date(2026, 10, 10)
        assert parse_ledger_date("10-10-2026") == date(2026, 10, 10)
        assert parse_ledger_date(datetime(2026, 10, 10, 21, 30)) == date(2026, 10, 10)
        assert parse_ledger_date(pd.Timestamp("2026-10-10")) == date(2026, 10, 10)
        assert parse_ledger_date(None) is None
        assert parse_ledger_date("-") is None
        assert parse_ledger_date("") is None



# ==============================================================================
# 6. Tests de Selector por Lotes y Filtro de Fechas
# ==============================================================================

class TestBatchSelectionAndDateFiltering:
    def test_parse_batch_selection_single(self):
        """Verifica la selección de un solo partido numérico."""
        indices, cmd = parse_batch_selection("1", 5)
        assert indices == [1]
        assert cmd is None

        indices, cmd = parse_batch_selection("4", 5)
        assert indices == [4]
        assert cmd is None

    def test_parse_batch_selection_comma_list(self):
        """Verifica la selección de múltiples partidos separados por comas."""
        indices, cmd = parse_batch_selection("1, 3, 5", 5)
        assert indices == [1, 3, 5]
        assert cmd is None

        indices, cmd = parse_batch_selection("1,3,5", 5)
        assert indices == [1, 3, 5]
        assert cmd is None

        # Deduplicación preservando orden
        indices, cmd = parse_batch_selection("2, 1, 2", 5)
        assert indices == [2, 1]
        assert cmd is None

    def test_parse_batch_selection_todos_keyword(self):
        """Verifica que 'todos' o 'all' seleccione todos los partidos listados."""
        indices, cmd = parse_batch_selection("todos", 4)
        assert indices == [1, 2, 3, 4]
        assert cmd is None

        indices, cmd = parse_batch_selection("TODOS", 3)
        assert indices == [1, 2, 3]
        assert cmd is None

        indices, cmd = parse_batch_selection("all", 2)
        assert indices == [1, 2]
        assert cmd is None

    def test_parse_batch_selection_navigation_commands(self):
        """Verifica los comandos rápidos de vista [M], [T], [H] y manual [0]."""
        indices, cmd = parse_batch_selection("m", 5)
        assert indices == []
        assert cmd == "tomorrow"

        indices, cmd = parse_batch_selection("M", 5)
        assert indices == []
        assert cmd == "tomorrow"

        indices, cmd = parse_batch_selection("t", 5)
        assert indices == []
        assert cmd == "all_view"

        indices, cmd = parse_batch_selection("T", 5)
        assert indices == []
        assert cmd == "all_view"

        indices, cmd = parse_batch_selection("h", 5)
        assert indices == []
        assert cmd == "today"

        indices, cmd = parse_batch_selection("H", 5)
        assert indices == []
        assert cmd == "today"

        indices, cmd = parse_batch_selection("0", 5)
        assert indices == [0]
        assert cmd is None

    def test_parse_batch_selection_errors(self):
        """Verifica el manejo de entradas inválidas o fuera de rango."""
        indices, cmd = parse_batch_selection("99", 5)
        assert indices == []
        assert cmd == "out_of_range"

        indices, cmd = parse_batch_selection("1, 8", 5)
        assert indices == []
        assert cmd == "out_of_range"

        indices, cmd = parse_batch_selection("abc", 5)
        assert indices == []
        assert cmd == "invalid"

        indices, cmd = parse_batch_selection("", 5)
        assert indices == []
        assert cmd == "empty"

    @patch("requests.get")
    def test_get_games_by_filter_categories(self, mock_get):
        """Verifica el filtrado de partidos por categorías 'today', 'tomorrow', 'upcoming', 'all'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "game_today",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "commence_time": "2026-10-10T23:00:00Z",  # Hoy (18:00 local UTC-5)
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.70}, {"name": "Miami Heat", "price": 2.20}]}]
                    }
                ]
            },
            {
                "id": "game_tomorrow",
                "home_team": "Dallas Mavericks",
                "away_team": "Phoenix Suns",
                "commence_time": "2026-10-11T23:00:00Z",  # Mañana
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Dallas Mavericks", "price": 1.85}, {"name": "Phoenix Suns", "price": 2.05}]}]
                    }
                ]
            },
            {
                "id": "game_upcoming",
                "home_team": "Denver Nuggets",
                "away_team": "Los Angeles Lakers",
                "commence_time": "2026-10-15T23:00:00Z",  # Próximos
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "Denver Nuggets", "price": 1.55}, {"name": "Los Angeles Lakers", "price": 2.50}]}]
                    }
                ]
            },
        ]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="valid_key")
        tz_bogota = timezone(timedelta(hours=-5))
        ref_now = datetime(2026, 10, 10, 17, 0, 0, tzinfo=timezone.utc)

        # Filtro HOY
        today_games = client.get_games_by_filter(
            filter_type="today",
            target_timezone=tz_bogota,
            reference_now_utc=ref_now,
        )
        assert len(today_games) == 1
        assert today_games[0]["game_id"] == "game_today"

        # Filtro MAÑANA
        tomorrow_games = client.get_games_by_filter(
            filter_type="tomorrow",
            target_timezone=tz_bogota,
            reference_now_utc=ref_now,
        )
        assert len(tomorrow_games) == 1
        assert tomorrow_games[0]["game_id"] == "game_tomorrow"

        # Filtro PRÓXIMOS
        upcoming_games = client.get_games_by_filter(
            filter_type="upcoming",
            target_timezone=tz_bogota,
            reference_now_utc=ref_now,
        )
        assert len(upcoming_games) == 1
        assert upcoming_games[0]["game_id"] == "game_upcoming"

        # Filtro TODOS
        all_games = client.get_games_by_filter(
            filter_type="all",
            target_timezone=tz_bogota,
            reference_now_utc=ref_now,
        )
        assert len(all_games) == 3

    @patch("requests.get")
    def test_group_games_by_date(self, mock_get):
        """Verifica la agrupación de partidos en diccionario con claves HOY, MAÑANA, PRÓXIMOS."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "g1",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "commence_time": "2026-10-10T23:00:00Z",
                "bookmakers": []
            },
            {
                "id": "g2",
                "home_team": "Dallas Mavericks",
                "away_team": "Phoenix Suns",
                "commence_time": "2026-10-11T23:00:00Z",
                "bookmakers": []
            },
        ]
        mock_get.return_value = mock_response

        client = LiveOddsClient(api_key="valid_key")
        tz_bogota = timezone(timedelta(hours=-5))
        ref_now = datetime(2026, 10, 10, 17, 0, 0, tzinfo=timezone.utc)

        grouped = client.group_games_by_date(
            target_timezone=tz_bogota,
            reference_now_utc=ref_now,
        )
        assert "HOY" in grouped
        assert "MAÑANA" in grouped
        assert "PRÓXIMOS" in grouped
        assert len(grouped["HOY"]) == 1
        assert len(grouped["MAÑANA"]) == 1
        assert len(grouped["PRÓXIMOS"]) == 0

    def test_cli_execute_batch_predictions_summary(self, temp_excel_path: Path):
        """Verifica la ejecución por lotes de partidos, registro en Excel y resumen consolidado."""
        mock_use_case = MagicMock()

        # Partido 1: +EV -> recomendado
        pred_rec = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.80,
            key_factors=["Fuerte local"],
            projected_margin=6.5,
            home_expected_value=0.15,
            away_expected_value=-0.20,
            kelly_stake_home=0.035,
            recommended_bet="Bet on Boston Celtics (Quarter Kelly: 3.5%)",
        )

        # Partido 2: -EV -> No bet
        pred_no_bet = PredictionResult(
            home_team="Dallas Mavericks",
            away_team="Phoenix Suns",
            home_win_prob=0.50,
            away_win_prob=0.45,
            confidence=0.50,
            key_factors=["Encuentro muy parejo"],
            projected_margin=0.5,
            home_expected_value=-0.15,
            away_expected_value=-0.23,
            recommended_bet="No bet",
        )

        mock_use_case.execute.side_effect = [pred_rec, pred_no_bet]

        controller = CLIController(predict_game=mock_use_case, excel_path=temp_excel_path)

        matches_batch = [
            {
                "home_id": 1610612738, "home_name": "Boston Celtics", "home_abbr": "BOS",
                "away_id": 1610612748, "away_name": "Miami Heat", "away_abbr": "MIA",
                "odds": Odds(home_decimal=1.65, away_decimal=2.35),
                "bookmaker": "Pinnacle",
                "game_id": "game_1",
                "is_pregame": True,
            },
            {
                "home_id": 1610612742, "home_name": "Dallas Mavericks", "home_abbr": "DAL",
                "away_id": 1610612756, "away_name": "Phoenix Suns", "away_abbr": "PHX",
                "odds": Odds(home_decimal=1.70, away_decimal=1.70),
                "bookmaker": "DraftKings",
                "game_id": "game_2",
                "is_pregame": True,
            },
        ]

        analyzed, recommended = controller.execute_batch(matches_batch, is_batch=True)

        assert analyzed == 2
        assert recommended == 1

        # Verificar que el archivo Excel contiene las 2 predicciones registradas (1 fila por partido)
        assert temp_excel_path.exists()
        df = load_ledger(temp_excel_path)
        assert len(df) == 2
        assert re.match(r"^\d{2}/\d{2}/\d{4}$", str(df.iloc[0]["Fecha"]))
        assert df.iloc[0]["Partido"] == "BOS - MIA"
        assert df.iloc[0]["Decision"] == "APOSTAR"
        assert df.iloc[0]["Marcador_Final"] == "-"
        assert df.iloc[0]["Estado"] == "PENDIENTE"

        match_2 = df[df["Partido"] == "DAL - PHX"]
        assert not match_2.empty
        assert re.match(r"^\d{2}/\d{2}/\d{4}$", str(match_2.iloc[0]["Fecha"]))
        assert match_2.iloc[0]["Decision"] == "NO APOSTAR"
        assert match_2.iloc[0]["Marcador_Final"] == "-"
        assert match_2.iloc[0]["Estado"] == "PENDIENTE"


# ==============================================================================
# 7. Tests de Sincronización de Zona Horaria Local y Formateo
# ==============================================================================

class TestLocalTimezoneAndExcelFormatting:
    def test_format_local_timestamp_structure(self):
        """Verifica que format_local_timestamp produzca exactamente DD/MM/YYYY HH:MM:SS."""
        ts = format_local_timestamp()
        assert re.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$", ts)

    def test_format_local_timestamp_utc_conversion(self):
        """Verifica que un datetime en UTC sea convertido a la hora local (America/Bogota, UTC-5)."""
        dt_utc = datetime(2026, 10, 11, 2, 30, 0, tzinfo=timezone.utc)
        # En UTC-5, esto corresponde al 10/10/2026 a las 21:30:00
        formatted = format_local_timestamp(dt_utc)
        assert formatted == "10/10/2026 21:30:00"

    def test_format_date_ddmmyyyy_iso_utc_evening_no_day_drift(self):
        """
        Verifica que un partido jugado a las 8:30 PM (20:30) hora local (01:30 UTC del día siguiente)
        NO sufra el salto prematuro de fecha al día siguiente en el Excel.
        """
        # 1:30 AM UTC del 11 de Octubre = 8:30 PM del 10 de Octubre en Bogotá (UTC-5)
        iso_commence_time = "2026-10-11T01:30:00Z"
        date_str = format_date_ddmmyyyy(iso_commence_time)
        assert date_str == "10/10/2026"

        # Con offset explícito +00:00
        iso_commence_time_offset = "2026-10-11T00:30:00+00:00"
        date_str_offset = format_date_ddmmyyyy(iso_commence_time_offset)
        assert date_str_offset == "10/10/2026"

    def test_format_date_ddmmyyyy_standard_formats(self):
        """Verifica que formatos estándar YYYY-MM-DD y DD/MM/YYYY se procesen limpiamente."""
        assert format_date_ddmmyyyy("2026-10-10") == "10/10/2026"
        assert format_date_ddmmyyyy("10/10/2026") == "10/10/2026"
        assert format_date_ddmmyyyy("10-10-2026") == "10/10/2026"

    def test_format_date_ddmmyyyy_fallback_local_now(self):
        """Verifica que en ausencia de fecha se use la fecha actual en la zona horaria local."""
        expected_today = get_local_now().strftime("%d/%m/%Y")
        assert format_date_ddmmyyyy(None) == expected_today
        assert format_date_ddmmyyyy("") == expected_today
        assert format_date_ddmmyyyy("-") == expected_today

    def test_empty_dataframe_metrics_summary_has_local_timestamp(self):
        """Verifica que la métrica 'Última Actualización' con DataFrame vacío use formato local."""
        df_empty = pd.DataFrame(columns=COLUMNS_LEDGER)
        metrics = calculate_metrics_summary(df_empty)
        metrics_dict = dict(zip(metrics["Métrica"], metrics["Valor"]))
        assert "Última Actualización" in metrics_dict
        ts = str(metrics_dict["Última Actualización"])
        assert re.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$", ts)


# ==============================================================================
# 8. Tests de Liquidación Manual Forzada (get_pending_bets & settle_bet_manually)
# ==============================================================================

class TestManualBetSettlement:
    def test_get_pending_bets_returns_structured_list(self, temp_excel_path: Path):
        """Verifica que get_pending_bets retorne la lista estructurada de partidos pendientes."""
        # 1. Sin archivo aún
        assert get_pending_bets(temp_excel_path) == []

        # 2. Registrar 2 partidos
        pred1 = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.80,
            key_factors=["Margen proyectado: +6.5 pts"],
            projected_margin=6.5,
            home_expected_value=0.25,
            away_expected_value=-0.20,
            recommended_bet="Boston Celtics (-110)",
        )
        odds1 = Odds(home_decimal=1.85, away_decimal=2.00)
        log_bet_prediction(pred1, odds=odds1, game_id="001", excel_path=temp_excel_path)

        pred2 = PredictionResult(
            home_team="Los Angeles Lakers",
            away_team="Golden State Warriors",
            home_win_prob=0.55,
            away_win_prob=0.45,
            confidence=0.60,
            key_factors=["Margen proyectado: +1.5 pts"],
            projected_margin=1.5,
            home_expected_value=-0.05,
            away_expected_value=-0.10,
            recommended_bet="No bet",
        )
        log_bet_prediction(pred2, odds=None, game_id="002", excel_path=temp_excel_path)

        pending = get_pending_bets(temp_excel_path)
        assert len(pending) == 2

        bet1 = pending[0]
        assert bet1["id"] == 1
        assert bet1["partido"] == "BOS - MIA"
        assert bet1["team_1"] == "BOS"
        assert bet1["team_2"] == "MIA"
        assert bet1["decision"] == "APOSTAR"
        assert "1.85" in bet1["cuota"]
        assert "BOS" in bet1["probabilidad"]

        bet2 = pending[1]
        assert bet2["id"] == 2
        assert bet2["partido"] == "LAL - GSW"
        assert bet2["decision"] == "NO APOSTAR"
        assert bet2["cuota"] == "NO DISPONIBLE"

    def test_settle_bet_manually_won_updates_rows_and_metrics(self, temp_excel_path: Path):
        """Verifica que settle_bet_manually liquide con acierto y actualice filas y métricas."""
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.75,
            away_win_prob=0.25,
            confidence=0.85,
            key_factors=["Margen proyectado: +8.0 pts"],
            projected_margin=8.0,
            home_expected_value=0.30,
            away_expected_value=-0.25,
            recommended_bet="Boston Celtics",
        )
        odds = Odds(home_decimal=1.90, away_decimal=1.90)
        log_bet_prediction(pred, odds=odds, game_id="001", excel_path=temp_excel_path)

        # Liquidar manualmente: BOS (112) vs MIA (98) -> BOS gana -> ACERTÓ
        res = settle_bet_manually(
            match_identifier=1,
            home_score=112,
            away_score=98,
            excel_path=temp_excel_path,
        )

        assert res["success"] is True
        assert res["status"] == "ACERTÓ"
        assert res["score"] == "112 - 98"
        assert res["margin"] == 14
        assert res["pnl"] == 0.90

        # Verificar hoja Registro_Apuestas (1 fila por partido)
        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        assert df.iloc[0]["Estado"] == "ACERTÓ"
        assert df.iloc[0]["Marcador_Final"] == "112 - 98"

        # Verificar hoja Resumen_Metricas
        wb = openpyxl.load_workbook(temp_excel_path, data_only=True)
        ws_m = wb[SHEET_METRICS]
        m_dict = {ws_m.cell(r, 1).value: ws_m.cell(r, 2).value for r in range(2, ws_m.max_row + 1)}
        assert m_dict["Total Encuentros Analizados"] == 1
        assert m_dict["Apuestas Ganadas"] == 1
        assert m_dict["Encuentros Pendientes"] == 0
        assert round(float(m_dict["P&L Neto Estimado (Unidades)"]), 2) == 0.90

    def test_settle_bet_manually_lost_by_name(self, temp_excel_path: Path):
        """Verifica liquidación manual de fallo usando nombre de partido ('PHI - NYK')."""
        pred = PredictionResult(
            home_team="Philadelphia 76ers",
            away_team="New York Knicks",
            home_win_prob=0.65,
            away_win_prob=0.35,
            confidence=0.70,
            key_factors=["Margen proyectado: +4.0 pts"],
            projected_margin=4.0,
            home_expected_value=0.20,
            away_expected_value=-0.15,
            recommended_bet="Philadelphia 76ers",
        )
        odds = Odds(home_decimal=1.80, away_decimal=2.05)
        log_bet_prediction(pred, odds=odds, game_id="003", excel_path=temp_excel_path)

        # PHI (95) vs NYK (104) -> NYK gana -> FALLÓ
        res = settle_bet_manually(
            match_identifier="PHI - NYK",
            home_score=95,
            away_score=104,
            excel_path=temp_excel_path,
        )

        assert res["success"] is True
        assert res["status"] == "FALLÓ"
        assert res["score"] == "95 - 104"
        assert res["margin"] == -9
        assert res["pnl"] == -1.0

        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        assert df.iloc[0]["Estado"] == "FALLÓ"
        assert df.iloc[0]["Marcador_Final"] == "95 - 104"

    def test_prompt_manual_settlement_cli_flow(self, temp_excel_path: Path):
        """Verifica el flujo interactivo prompt_manual_settlement del CLIController."""
        pred = PredictionResult(
            home_team="Denver Nuggets",
            away_team="Phoenix Suns",
            home_win_prob=0.60,
            away_win_prob=0.40,
            confidence=0.70,
            key_factors=["Margen: +3.0 pts"],
            projected_margin=3.0,
            home_expected_value=0.15,
            away_expected_value=-0.10,
            recommended_bet="Denver Nuggets",
        )
        odds = Odds(home_decimal=1.95, away_decimal=1.95)
        log_bet_prediction(pred, odds=odds, game_id="004", excel_path=temp_excel_path)

        mock_use_case = MagicMock()
        controller = CLIController(predict_game=mock_use_case, excel_path=temp_excel_path)

        # Simular entradas del usuario: Seleccionar partido 1, Puntos Local 110, Puntos Visitante 102
        with patch("builtins.input", side_effect=["1", "110", "102"]):
            success = controller.prompt_manual_settlement()

        assert success is True
        df = load_ledger(temp_excel_path)
        assert len(df) == 1
        assert df.iloc[0]["Estado"] == "ACERTÓ"
        assert df.iloc[0]["Marcador_Final"] == "110 - 102"





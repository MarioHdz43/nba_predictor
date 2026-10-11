# -*- coding: utf-8 -*-
"""
infrastructure/reporting/settle_bets.py
---------------------------------------
Módulo de Liquidación y Auto-Reconciliación para el libro mayor en Excel de 8 columnas.
Cruza los bloques de pronósticos pendientes contra marcadores oficiales de partidos finalizados
consultando stats.nba.com (ScoreboardV2) y SQLite (historical_games).
"""

from __future__ import annotations
from datetime import date, datetime, timedelta
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import structlog

from domain.entities import Odds
from infrastructure.config import config
from infrastructure.odds.live_odds_client import canonicalize_team
from infrastructure.reporting.bet_tracker_excel import (
    COLUMNS_LEDGER,
    SHEET_BETS,
    get_excel_path,
    get_local_now,
    load_ledger,
    save_ledger_workbook,
)

logger = structlog.get_logger()

# Mapeo canónico de franquicias NBA (ID numérico <-> Abreviatura)
NBA_TEAM_NAME_TO_ID: Dict[str, int] = {
    "ATLANTA HAWKS": 1610612737, "HAWKS": 1610612737, "ATL": 1610612737,
    "BOSTON CELTICS": 1610612738, "CELTICS": 1610612738, "BOS": 1610612738,
    "BROOKLYN NETS": 1610612751, "NETS": 1610612751, "BKN": 1610612751,
    "CHARLOTTE HORNETS": 1610612766, "HORNETS": 1610612766, "CHA": 1610612766,
    "CHICAGO BULLS": 1610612741, "BULLS": 1610612741, "CHI": 1610612741,
    "CLEVELAND CAVALIERS": 1610612739, "CAVALIERS": 1610612739, "CLE": 1610612739,
    "DALLAS MAVERICKS": 1610612742, "MAVERICKS": 1610612742, "DAL": 1610612742,
    "DENVER NUGGETS": 1610612743, "NUGGETS": 1610612743, "DEN": 1610612743,
    "DETROIT PISTONS": 1610612765, "PISTONS": 1610612765, "DET": 1610612765,
    "GOLDEN STATE WARRIORS": 1610612744, "WARRIORS": 1610612744, "GSW": 1610612744,
    "HOUSTON ROCKETS": 1610612745, "ROCKETS": 1610612745, "HOU": 1610612745,
    "INDIANA PACERS": 1610612754, "PACERS": 1610612754, "IND": 1610612754,
    "LOS ANGELES CLIPPERS": 1610612746, "CLIPPERS": 1610612746, "LAC": 1610612746, "LA CLIPPERS": 1610612746,
    "LOS ANGELES LAKERS": 1610612747, "LAKERS": 1610612747, "LAL": 1610612747, "LA LAKERS": 1610612747,
    "MEMPHIS GRIZZLIES": 1610612763, "GRIZZLIES": 1610612763, "MEM": 1610612763,
    "MIAMI HEAT": 1610612748, "HEAT": 1610612748, "MIA": 1610612748,
    "MILWAUKEE BUCKS": 1610612749, "BUCKS": 1610612749, "MIL": 1610612749,
    "MINNESOTA TIMBERWOLVES": 1610612750, "TIMBERWOLVES": 1610612750, "MIN": 1610612750,
    "NEW ORLEANS PELICANS": 1610612740, "PELICANS": 1610612740, "NOP": 1610612740,
    "NEW YORK KNICKS": 1610612752, "KNICKS": 1610612752, "NYK": 1610612752,
    "OKLAHOMA CITY THUNDER": 1610612760, "THUNDER": 1610612760, "OKC": 1610612760,
    "ORLANDO MAGIC": 1610612753, "MAGIC": 1610612753, "ORL": 1610612753,
    "PHILADELPHIA 76ERS": 1610612755, "76ERS": 1610612755, "PHI": 1610612755,
    "PHOENIX SUNS": 1610612756, "SUNS": 1610612756, "PHX": 1610612756,
    "PORTLAND TRAIL BLAZERS": 1610612757, "TRAIL BLAZERS": 1610612757, "POR": 1610612757,
    "SACRAMENTO KINGS": 1610612758, "KINGS": 1610612758, "SAC": 1610612758,
    "SAN ANTONIO SPURS": 1610612759, "SPURS": 1610612759, "SAS": 1610612759,
    "TORONTO RAPTORS": 1610612761, "RAPTORS": 1610612761, "TOR": 1610612761,
    "UTAH JAZZ": 1610612762, "JAZZ": 1610612762, "UTA": 1610612762,
    "WASHINGTON WIZARDS": 1610612764, "WIZARDS": 1610612764, "WAS": 1610612764,
}

ID_TO_ABBR: Dict[int, str] = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS",
}


def _resolve_team_id(team_str: Any) -> Optional[int]:
    """Resuelve el ID oficial de un equipo a partir de su nombre o abreviatura."""
    if not team_str:
        return None
    s = str(team_str).strip().upper()
    if s.isdigit():
        return int(s)
    return NBA_TEAM_NAME_TO_ID.get(s)


def _resolve_team_abbr(team_str: Any) -> Optional[str]:
    """Resuelve la abreviatura canónica oficial de 3 letras de un equipo."""
    if not team_str:
        return None
    s = str(team_str).strip()
    c = canonicalize_team(s)
    if c:
        return c.upper()
    tid = _resolve_team_id(s)
    if tid and tid in ID_TO_ABBR:
        return ID_TO_ABBR[tid]
    return s[:3].upper() if len(s) >= 3 else None


def _parse_score_text(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """Extrae marcador local y visitante a partir de texto como '110-105' o '110 - 105'."""
    if not text:
        return None
    m = re.search(r"(\d{2,3})\s*[-–:]\s*(\d{2,3})", str(text))
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except ValueError:
            pass
    return None


def parse_ledger_date(date_val: Any) -> Optional[date]:
    """
    Parsea el valor de fecha del Excel a datetime.date.
    Soporta formatos DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, datetime y pd.Timestamp.
    """
    if not date_val or str(date_val).strip() in ("-", "None", "nan", ""):
        return None
    if isinstance(date_val, (datetime, pd.Timestamp)):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    s = str(date_val).strip()
    m_slash = re.match(r"^(\d{2})/(\d{2})/(\d{4})", s)
    if m_slash:
        dd, mm, yyyy = m_slash.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except ValueError:
            pass
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m_iso:
        yyyy, mm, dd = m_iso.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except ValueError:
            pass
    m_dash = re.match(r"^(\d{2})-(\d{2})-(\d{4})", s)
    if m_dash:
        dd, mm, yyyy = m_dash.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except ValueError:
            pass
    return None


def _load_historical_results_from_db(
    db_path: Optional[Union[str, Path]],
) -> Tuple[
    Dict[str, Tuple[int, int]],
    Dict[Tuple[int, int, Optional[str]], Tuple[int, int]],
    Dict[Tuple[str, str], Tuple[int, int]],
]:
    """
    Lee partidos terminados desde SQLite.
    Retorna:
    - results_by_id: Dict[game_id, (home_score, away_score)]
    - results_by_matchup: Dict[(home_id, away_id, date), (home_score, away_score)]
    - results_by_abbr: Dict[(home_abbr, away_abbr), (home_score, away_score)]
    """
    results_by_id: Dict[str, Tuple[int, int]] = {}
    results_by_matchup: Dict[Tuple[int, int, Optional[str]], Tuple[int, int]] = {}
    results_by_abbr: Dict[Tuple[str, str], Tuple[int, int]] = {}

    target_db = Path(db_path) if db_path else Path(getattr(config, "database_path", "nba_predictor.db"))
    if not target_db.exists():
        logger.debug("historical-db-not-found", path=str(target_db))
        return results_by_id, results_by_matchup, results_by_abbr

    try:
        conn = sqlite3.connect(str(target_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Consultar historical_games
        try:
            cursor.execute("""
                SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score
                FROM historical_games
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            """)
            for row in cursor.fetchall():
                gid = str(row["game_id"])
                h_score = int(row["home_score"])
                a_score = int(row["away_score"])
                h_id = int(row["home_team_id"])
                a_id = int(row["away_team_id"])
                g_date = str(row["game_date"]).strip()

                pair = (h_score, a_score)
                results_by_id[gid] = pair
                results_by_matchup[(h_id, a_id, g_date)] = pair
                results_by_matchup[(h_id, a_id, None)] = pair

                h_abbr = ID_TO_ABBR.get(h_id)
                a_abbr = ID_TO_ABBR.get(a_id)
                if h_abbr and a_abbr:
                    results_by_abbr[(h_abbr, a_abbr)] = pair
        except sqlite3.OperationalError:
            pass

        # 2. Consultar predictions con actual_result
        try:
            cursor.execute("""
                SELECT id, home_team, away_team, actual_result, predicted_at
                FROM predictions
                WHERE actual_result IS NOT NULL
            """)
            for row in cursor.fetchall():
                score_pair = _parse_score_text(row["actual_result"])
                if score_pair:
                    pred_id = str(row["id"])
                    results_by_id[pred_id] = score_pair

                    h_id = _resolve_team_id(row["home_team"])
                    a_id = _resolve_team_id(row["away_team"])
                    h_abbr = _resolve_team_abbr(row["home_team"])
                    a_abbr = _resolve_team_abbr(row["away_team"])
                    if h_id and a_id:
                        p_date = str(row["predicted_at"])[:10]
                        results_by_matchup[(h_id, a_id, p_date)] = score_pair
                        results_by_matchup[(h_id, a_id, None)] = score_pair
                    if h_abbr and a_abbr:
                        results_by_abbr[(h_abbr, a_abbr)] = score_pair
        except sqlite3.OperationalError:
            pass

        conn.close()
    except Exception as exc:
        logger.warning(f"db-results-query-error: {exc}")

    return results_by_id, results_by_matchup, results_by_abbr


def settle_pending_bets(
    db_path: Optional[Union[str, Path]] = None,
    excel_path: Optional[Union[str, Path]] = None,
    fallback_results: Optional[List[Dict[str, Any]]] = None,
    api_results: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
    closing_odds_map: Optional[Dict[str, Odds]] = None,
    local_today: Optional[date] = None,
    print_logs: bool = True,
) -> Dict[str, Any]:
    """
    Liquida encuentros pendientes en la bitácora Excel de 7 columnas (1 fila por evento).
    Aplica cruce bidireccional de equipos y evalúa partidos con game_date <= local_today.
    Actualiza directamente en la fila del encuentro:
      - Marcador_Final = score_text ("112 - 104")
      - Estado = "ACERTÓ" / "FALLÓ" / "PUSH"
    Retorna resumen cuantitativo de la operación.
    """
    target_excel = get_excel_path(excel_path)
    if not target_excel.exists():
        return {
            "settled": 0,
            "won": 0,
            "lost": 0,
            "push": 0,
            "total_pnl": 0.0,
            "pending_remaining": 0,
            "excel_path": str(target_excel),
            "message": "Archivo Excel no existe.",
            "details": [],
        }

    ledger_df = load_ledger(target_excel)
    if ledger_df.empty:
        return {
            "settled": 0,
            "won": 0,
            "lost": 0,
            "push": 0,
            "total_pnl": 0.0,
            "pending_remaining": 0,
            "excel_path": str(target_excel),
            "message": "Bitácora vacía.",
            "details": [],
        }

    ledger_df = ledger_df.astype(object)

    # Cargar resultados desde SQLite
    results_by_id, results_by_matchup, results_by_abbr = _load_historical_results_from_db(db_path)

    # Incorporar marcadores en vivo desde la API de la NBA
    if api_results:
        for (h_a, a_a), pts_pair in api_results.items():
            results_by_abbr[(str(h_a).upper(), str(a_a).upper())] = pts_pair

    # Incorporar fallback_results si se proveyeron
    if fallback_results:
        for item in fallback_results:
            gid = str(item.get("game_id", "")).strip()
            h_score = item.get("home_score")
            away_score = item.get("away_score")
            if h_score is not None and away_score is not None:
                pair = (int(h_score), int(away_score))
                if gid:
                    results_by_id[gid] = pair

                h_t = item.get("home_team")
                a_t = item.get("away_team")
                h_abbr = _resolve_team_abbr(h_t) if h_t else None
                a_abbr = _resolve_team_abbr(a_t) if a_t else None

                if h_abbr and a_abbr:
                    results_by_abbr[(h_abbr, a_abbr)] = pair

                h_id = _resolve_team_id(h_t) if h_t else None
                a_id = _resolve_team_id(a_t) if a_t else None
                g_date = str(item.get("game_date", "")).strip() or None
                if h_id and a_id:
                    results_by_matchup[(h_id, a_id, g_date)] = pair
                    results_by_matchup[(h_id, a_id, None)] = pair

    settled_count = 0
    won_count = 0
    lost_count = 0
    push_count = 0
    batch_pnl = 0.0
    details = []

    # Recorrer fila por fila en el esquema tabular de 1 fila por evento
    n_rows = len(ledger_df)
    for idx_1 in range(n_rows):
        row_1 = ledger_df.iloc[idx_1]
        partido_raw = str(row_1.get("Partido") or "").strip()
        status_1 = str(row_1.get("Estado") or "").strip().upper()

        # Fila vacía o de datos no válidos
        if not partido_raw or partido_raw in ("-", "None", "nan"):
            continue

        if status_1 in ("PENDIENTE", "", "NONE"):
            # 1. Filtro de fecha: game_date <= local_today si local_today se proporciona
            game_date_obj = parse_ledger_date(row_1.get("Fecha"))
            if local_today is not None and game_date_obj is not None:
                if game_date_obj > local_today:
                    continue

            # 2. Extraer equipos desde la columna Partido (ej. 'PHI - BOS')
            parts = [p.strip() for p in re.split(r"\s*[-–vs.]+\s*", partido_raw) if p.strip()]
            if len(parts) >= 2:
                team_1_raw, team_2_raw = parts[0], parts[1]
            else:
                team_1_raw, team_2_raw = partido_raw, ""

            t1_abbr = _resolve_team_abbr(team_1_raw) or team_1_raw.upper()
            t2_abbr = _resolve_team_abbr(team_2_raw) or team_2_raw.upper()
            t1_id = _resolve_team_id(team_1_raw)
            t2_id = _resolve_team_id(team_2_raw)

            score_team_1: Optional[int] = None
            score_team_2: Optional[int] = None

            # 3. Cruce Bidireccional de Equipos:
            # (home == t1_abbr and away == t2_abbr) O (home == t2_abbr and away == t1_abbr)

            # Paso A: Prioridad 1 -> api_results (Marcadores en vivo oficiales de hoy/recientes)
            if api_results:
                if (t1_abbr, t2_abbr) in api_results:
                    score_team_1, score_team_2 = api_results[(t1_abbr, t2_abbr)]
                elif (t2_abbr, t1_abbr) in api_results:
                    score_team_2, score_team_1 = api_results[(t2_abbr, t1_abbr)]

            # Paso B: Prioridad 2 -> fallback_results (suministrados explícitamente)
            if score_team_1 is None and fallback_results:
                for fb in fallback_results:
                    fb_gid = str(fb.get("game_id", "")).strip()
                    fb_h = _resolve_team_abbr(fb.get("home_team")) if fb.get("home_team") else ""
                    fb_a = _resolve_team_abbr(fb.get("away_team")) if fb.get("away_team") else ""
                    h_sc = fb.get("home_score")
                    a_sc = fb.get("away_score")
                    if h_sc is not None and a_sc is not None:
                        if fb_h == t1_abbr and fb_a == t2_abbr:
                            score_team_1 = int(h_sc)
                            score_team_2 = int(a_sc)
                            break
                        elif fb_h == t2_abbr and fb_a == t1_abbr:
                            score_team_1 = int(a_sc)
                            score_team_2 = int(h_sc)
                            break
                    if fb_gid and fb_gid in results_by_id and len(fallback_results) == 1:
                        score_team_1, score_team_2 = results_by_id[fb_gid]
                        break

            # Paso C: Prioridad 3 -> DB SQLite con coincidencia de fecha exacta
            if score_team_1 is None and (t1_id and t2_id):
                date_candidates = []
                if game_date_obj:
                    date_candidates.append(game_date_obj.strftime("%Y-%m-%d"))
                for d_k in date_candidates:
                    if (t1_id, t2_id, d_k) in results_by_matchup:
                        score_team_1, score_team_2 = results_by_matchup[(t1_id, t2_id, d_k)]
                        break
                    elif (t2_id, t1_id, d_k) in results_by_matchup:
                        score_team_2, score_team_1 = results_by_matchup[(t2_id, t1_id, d_k)]
                        break

            # Paso D: Fallback si se especificó una base de datos personalizada db_path
            if score_team_1 is None and db_path:
                if (t1_abbr, t2_abbr) in results_by_abbr:
                    score_team_1, score_team_2 = results_by_abbr[(t1_abbr, t2_abbr)]
                elif (t2_abbr, t1_abbr) in results_by_abbr:
                    score_team_2, score_team_1 = results_by_abbr[(t2_abbr, t1_abbr)]

            if score_team_1 is not None and score_team_2 is not None:
                score_text = f"{score_team_1} - {score_team_2}"
                real_margin = score_team_1 - score_team_2

                # Determinar si el pronóstico del modelo acertó
                if score_team_1 == score_team_2:
                    verificacion = "PUSH"
                else:
                    real_winner = t1_abbr if score_team_1 > score_team_2 else t2_abbr
                    prob_str = str(row_1.get("Probabilidad") or "").strip().upper()
                    if t2_abbr in prob_str:
                        proj_winner = t2_abbr
                    elif t1_abbr in prob_str:
                        proj_winner = t1_abbr
                    else:
                        proj_winner = t1_abbr

                    verificacion = "ACERTÓ" if (real_winner == proj_winner) else "FALLÓ"

                # Cálculo de P&L para apuestas
                decision = str(row_1.get("Decision") or "").strip().upper()
                cuota_raw = str(row_1.get("Cuota") or "1.90").replace(",", ".").strip()
                match_cuota = re.search(r"(\d+(?:\.\d+)?)", cuota_raw)
                if match_cuota:
                    try:
                        cuota = float(match_cuota.group(1))
                    except ValueError:
                        cuota = 1.90
                else:
                    cuota = 1.90

                pnl = 0.0
                if decision == "APOSTAR":
                    if verificacion == "ACERTÓ":
                        pnl = round(1.0 * (cuota - 1.0), 4)
                        won_count += 1
                    elif verificacion == "FALLÓ":
                        pnl = -1.0
                        lost_count += 1
                    else:
                        pnl = 0.0
                        push_count += 1
                else:
                    pnl = 0.0
                    if verificacion == "ACERTÓ":
                        won_count += 1
                    elif verificacion == "FALLÓ":
                        lost_count += 1
                    else:
                        push_count += 1

                batch_pnl += pnl
                settled_count += 1

                # Actualizar Fila única del encuentro con marcador real y estado verificado
                ledger_df.at[idx_1, "Marcador_Final"] = score_text
                ledger_df.at[idx_1, "Estado"] = verificacion

                if print_logs:
                    print(f"[Auto-Liquidación] Partido '{partido_raw}' finalizado (Marcador: {score_text}) -> Liquidado como {verificacion}.")

                details.append({
                    "matchup": f"{t1_abbr} vs {t2_abbr}",
                    "partido": partido_raw,
                    "score": score_text,
                    "margin": real_margin,
                    "status": verificacion,
                    "decision": decision,
                    "pnl": pnl,
                })

    if settled_count > 0:
        save_ledger_workbook(ledger_df, target_excel)
        logger.info(
            "games-settled",
            settled=settled_count,
            won=won_count,
            lost=lost_count,
            push=push_count,
            total_pnl=round(batch_pnl, 4),
        )

    # Contar cuántos encuentros siguen pendientes
    mask_pending = (
        ledger_df["Partido"].notna()
        & (ledger_df["Partido"].astype(str).str.strip() != "")
        & (ledger_df["Partido"].astype(str).str.strip() != "-")
        & (ledger_df["Estado"].astype(str).str.strip().str.upper() == "PENDIENTE")
    )
    rem_pending = int(mask_pending.sum())

    return {
        "settled": settled_count,
        "won": won_count,
        "lost": lost_count,
        "push": push_count,
        "total_pnl": round(batch_pnl, 4),
        "pending_remaining": rem_pending,
        "excel_path": str(target_excel),
        "details": details,
    }


def auto_reconcile_pending_bets(
    excel_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    fallback_results: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Auto-liquidación automática y transparente al iniciar la aplicación.
    Verifica si existen partidos con Estado == 'PENDIENTE'. Si los hay:
      - Imprime: [Auto-Liquidación] Verificando X partido(s) pendiente(s)...
      - Modifica el filtro de fecha para evaluar partidos con game_date <= local_today.
      - Consulta ScoreboardV2 tanto para la fecha del partido como para la fecha local actual.
      - Si el encuentro tiene estado oficial final (GAME_STATUS_ID == 3 o Final), lo liquida inmediatamente.
      - Imprime: [Auto-Liquidación] Partido 'EQUIPO1 - EQUIPO2' finalizado (Marcador: X - Y) -> Liquidado como ACERTÓ/FALLÓ.
      - Si partidos quedan sin finalizar: [Auto-Liquidación] X partido(s) aún no cuentan con marcador final oficial en NBA API.
    Si no hay partidos pendientes, finaliza inmediatamente sin demoras de red.
    """
    target_excel = get_excel_path(excel_path)
    if not target_excel.exists():
        return 0

    try:
        df = load_ledger(target_excel)
        if df.empty or "Estado" not in df.columns:
            return 0

        pending_mask = (
            df["Partido"].notna()
            & (df["Partido"].astype(str).str.strip() != "")
            & (df["Partido"].astype(str).str.strip() != "-")
            & (df["Estado"].astype(str).str.strip().str.upper() == "PENDIENTE")
        )
        pending_rows = df[pending_mask]
        pending_count = len(pending_rows)
        if pending_count == 0:
            return 0

        # Log informativo inicial
        print(f"\n[Auto-Liquidación] Verificando {pending_count} partido(s) pendiente(s)...")

        local_today = get_local_now().date()

        dates_to_query = set()
        dates_to_query.add(local_today.strftime("%Y-%m-%d"))
        dates_to_query.add((local_today - timedelta(days=1)).strftime("%Y-%m-%d"))

        for _, r in pending_rows.iterrows():
            p_date = parse_ledger_date(r.get("Fecha"))
            if p_date and p_date <= local_today:
                dates_to_query.add(p_date.strftime("%Y-%m-%d"))

        sorted_dates = sorted(list(dates_to_query))
        date_str_display = sorted_dates[0] if len(sorted_dates) == 1 else ", ".join(sorted_dates)
        print(f"[Auto-Liquidación] Consultando marcadores vía ESPN/NBA Network para fecha {date_str_display}...")

        api_results = {}
        try:
            from infrastructure.nba_api_client import fetch_scoreboard_results
            api_results = fetch_scoreboard_results(game_date=sorted_dates)
        except Exception as exc:
            logger.debug(f"auto-reconcile-fetch-scoreboard-error: {exc}")

        res = settle_pending_bets(
            db_path=db_path,
            excel_path=target_excel,
            fallback_results=fallback_results,
            api_results=api_results,
            local_today=local_today,
            print_logs=True,
        )

        rem = res.get("pending_remaining", 0)
        if rem > 0:
            print(f"[Auto-Liquidación] {rem} partido(s) aún no cuentan con marcador final oficial en NBA API.")

        return res.get("settled", 0)
    except Exception as exc:
        logger.warning(f"auto-reconcile-error: {exc}")
        return 0


def get_pending_bets(
    excel_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene la lista estructurada de encuentros con estado 'PENDIENTE' registrados en el Excel.
    Retorna lista de diccionarios con metadatos del partido para visualización y selección.
    """
    target_excel = get_excel_path(excel_path)
    if not target_excel.exists():
        return []

    try:
        df = load_ledger(target_excel)
        if df.empty or "Estado" not in df.columns:
            return []

        pending_list: List[Dict[str, Any]] = []
        pending_num = 1

        for idx, row in df.iterrows():
            partido_str = str(row.get("Partido") or "").strip()
            estado_str = str(row.get("Estado") or "").strip().upper()

            if partido_str and partido_str != "-" and estado_str == "PENDIENTE":
                parts = [p.strip() for p in re.split(r"\s*[-–vs.]+\s*", partido_str) if p.strip()]
                t1 = parts[0] if len(parts) >= 1 else partido_str
                t2 = parts[1] if len(parts) >= 2 else ""

                pending_list.append({
                    "id": pending_num,
                    "row_idx": idx,
                    "fecha": str(row.get("Fecha") or "-"),
                    "partido": partido_str,
                    "cuota": str(row.get("Cuota") or "NO DISPONIBLE"),
                    "probabilidad": str(row.get("Probabilidad") or "-"),
                    "decision": str(row.get("Decision") or "NO APOSTAR"),
                    "team_1": t1,
                    "team_2": t2,
                })
                pending_num += 1

        return pending_list
    except Exception as exc:
        logger.warning(f"get-pending-bets-error: {exc}")
        return []


def settle_bet_manually(
    match_identifier: Union[int, str],
    home_score: int,
    away_score: int,
    excel_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Liquida manualmente un partido pendiente en el Excel ingresando el marcador final real.
    - match_identifier: puede ser el ID numérico de la lista de pendientes (1, 2, ...),
      el índice de fila en el DataFrame, o el nombre del partido (ej. 'PHI - BOS').
    - home_score: puntos anotados por el equipo local (primer equipo del emparejamiento).
    - away_score: puntos anotados por el equipo visitante (segundo equipo del emparejamiento).

    Actualiza Fila 1 (Estado: ACERTÓ/FALLÓ/PUSH) y Fila 2 (Marcador_Final: 'PTS_H - PTS_A').
    Recalcula automáticamente la hoja 'Resumen_Metricas' y guarda el archivo Excel.
    """
    target_excel = get_excel_path(excel_path)
    if not target_excel.exists():
        return {"success": False, "message": "Archivo Excel no encontrado."}

    df = load_ledger(target_excel)
    if df.empty or "Estado" not in df.columns:
        return {"success": False, "message": "El libro mayor está vacío."}

    df = df.astype(object)

    pending_bets = get_pending_bets(target_excel)
    if not pending_bets:
        return {"success": False, "message": "No hay partidos pendientes para liquidar."}

    target_bet: Optional[Dict[str, Any]] = None

    if isinstance(match_identifier, int):
        for p in pending_bets:
            if p["id"] == match_identifier or p["row_idx"] == match_identifier:
                target_bet = p
                break
    else:
        m_str = str(match_identifier).strip().upper()
        for p in pending_bets:
            if p["partido"].upper() == m_str:
                target_bet = p
                break
            if p["team_1"].upper() in m_str and p["team_2"].upper() in m_str:
                target_bet = p
                break

    if target_bet is None:
        return {"success": False, "message": f"No se encontró el partido pendiente '{match_identifier}'."}

    idx_1 = target_bet["row_idx"]

    partido_label = target_bet["partido"]
    t1_abbr = target_bet["team_1"]
    t2_abbr = target_bet["team_2"]
    prob_str = target_bet["probabilidad"].upper()
    decision = target_bet["decision"].upper()
    cuota_raw = str(target_bet["cuota"]).replace(",", ".").strip()
    match_cuota = re.search(r"(\d+(?:\.\d+)?)", cuota_raw)
    if match_cuota:
        try:
            cuota_val = float(match_cuota.group(1))
        except ValueError:
            cuota_val = 1.90
    else:
        cuota_val = 1.90

    score_text = f"{home_score} - {away_score}"
    real_margin = home_score - away_score

    # Determinar verificación
    if home_score == away_score:
        verificacion = "PUSH"
    else:
        real_winner = t1_abbr if home_score > away_score else t2_abbr
        if t2_abbr in prob_str:
            proj_winner = t2_abbr
        elif t1_abbr in prob_str:
            proj_winner = t1_abbr
        else:
            proj_winner = t1_abbr

        verificacion = "ACERTÓ" if (real_winner == proj_winner) else "FALLÓ"

    # P&L
    pnl = 0.0
    if decision == "APOSTAR":
        if verificacion == "ACERTÓ":
            pnl = round(1.0 * (cuota_val - 1.0), 4)
        elif verificacion == "FALLÓ":
            pnl = -1.0
        else:
            pnl = 0.0
    else:
        pnl = 0.0

    # Actualizar fila única del encuentro
    df.at[idx_1, "Marcador_Final"] = score_text
    df.at[idx_1, "Estado"] = verificacion

    # Guardar y recalcular Resumen_Metricas
    save_ledger_workbook(df, target_excel)

    logger.info(
        "manual-bet-settled",
        partido=partido_label,
        marcador=score_text,
        estado=verificacion,
        pnl=pnl,
    )

    return {
        "success": True,
        "partido": partido_label,
        "score": score_text,
        "margin": real_margin,
        "status": verificacion,
        "decision": decision,
        "pnl": pnl,
        "excel_path": str(target_excel),
    }



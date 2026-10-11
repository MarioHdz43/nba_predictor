# -*- coding: utf-8 -*-
"""
infrastructure/nba_api_client.py
---------------------------------
Implementación concreta de TeamDataProvider y motor de marcadores en tiempo real.
Incorpora una Arquitectura de Liquidación en Cascada (Fallback Cascade) resiliente:
- Tier 1: API Pública de ESPN (alta disponibilidad, sin bloqueos ni caídas HTTP 500).
- Tier 2: NBA Live CDN Edge (todaysScoreboard_v2.json).
- Tier 3: The Odds API (/scores) y fallback seguro a stats.nba.com (ScoreboardV2).
"""

from __future__ import annotations
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from domain.entities import PlayerStat, Team, TeamStats
from domain.repositories import TeamDataProvider

logger = logging.getLogger(__name__)

# Columnas que esperamos de la API para estadísticas de equipo
_STAT_COLS = ["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "STL", "BLK", "PLUS_MINUS"]

# Timeouts explícitos (conexión, lectura) para no bloquear el hilo principal
DEFAULT_TIMEOUT: Tuple[float, float] = (5.0, 15.0)

# ---------------------------------------------------------------------------
# Diccionario de Normalización Canónica de Franquicias (30 Equipos NBA)
# ---------------------------------------------------------------------------
TEAM_CANONICAL_MAP: Dict[str, str] = {
    # Variantes ESPN, históricas y coloquiales
    "GS": "GSW",
    "PHO": "PHX",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "NJ": "BKN",
    "WSH": "WAS",
    "UTAH": "UTA",
    "CHO": "CHA",
    "BRK": "BKN",
    "NOH": "NOP",
    "SEA": "OKC",

    # Nombres oficiales completos (30 franquicias)
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",

    # Apodos de franquicias
    "HAWKS": "ATL",
    "CELTICS": "BOS",
    "NETS": "BKN",
    "HORNETS": "CHA",
    "BULLS": "CHI",
    "CAVALIERS": "CLE",
    "CAVS": "CLE",
    "MAVERICKS": "DAL",
    "MAVS": "DAL",
    "NUGGETS": "DEN",
    "PISTONS": "DET",
    "WARRIORS": "GSW",
    "ROCKETS": "HOU",
    "PACERS": "IND",
    "CLIPPERS": "LAC",
    "LAKERS": "LAL",
    "GRIZZLIES": "MEM",
    "HEAT": "MIA",
    "BUCKS": "MIL",
    "TIMBERWOLVES": "MIN",
    "WOLVES": "MIN",
    "PELICANS": "NOP",
    "KNICKS": "NYK",
    "THUNDER": "OKC",
    "MAGIC": "ORL",
    "76ERS": "PHI",
    "SIXERS": "PHI",
    "SUNS": "PHX",
    "TRAIL BLAZERS": "POR",
    "BLAZERS": "POR",
    "KINGS": "SAC",
    "SPURS": "SAS",
    "RAPTORS": "TOR",
    "JAZZ": "UTA",
    "WIZARDS": "WAS",

    # Ciudades
    "ATLANTA": "ATL",
    "BOSTON": "BOS",
    "BROOKLYN": "BKN",
    "CHARLOTTE": "CHA",
    "CHICAGO": "CHI",
    "CLEVELAND": "CLE",
    "DALLAS": "DAL",
    "DENVER": "DEN",
    "DETROIT": "DET",
    "GOLDEN STATE": "GSW",
    "HOUSTON": "HOU",
    "INDIANA": "IND",
    "MEMPHIS": "MEM",
    "MIAMI": "MIA",
    "MILWAUKEE": "MIL",
    "MINNESOTA": "MIN",
    "NEW ORLEANS": "NOP",
    "NEW YORK": "NYK",
    "OKLAHOMA CITY": "OKC",
    "ORLANDO": "ORL",
    "PHILADELPHIA": "PHI",
    "PHOENIX": "PHX",
    "PORTLAND": "POR",
    "SACRAMENTO": "SAC",
    "SAN ANTONIO": "SAS",
    "TORONTO": "TOR",
    "WASHINGTON": "WAS",

    # Tricodes oficiales (identidad)
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "CHA": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NYK": "NYK",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHX": "PHX", "POR": "POR",
    "SAC": "SAC", "SAS": "SAS", "TOR": "TOR", "UTA": "UTA", "WAS": "WAS",
}


def standardize_team_code(raw_name: str) -> str:
    """
    Estandariza cualquier variante de nombre o abreviatura al código tricodificado oficial (3 letras).
    Ejemplos: 'GS' -> 'GSW', 'Boston Celtics' -> 'BOS', 'PHO' -> 'PHX', 'UTAH' -> 'UTA'.
    """
    if not raw_name:
        return ""
    val = str(raw_name).strip()
    u_val = val.upper()
    if u_val in TEAM_CANONICAL_MAP:
        return TEAM_CANONICAL_MAP[u_val]

    # Limpieza de caracteres de puntuación (puntos, guiones)
    clean_u = re.sub(r"[^A-Z0-9\s]", "", u_val).strip()
    if clean_u in TEAM_CANONICAL_MAP:
        return TEAM_CANONICAL_MAP[clean_u]

    # Si es un código alfabético de 3 letras ya en mayúsculas
    if len(clean_u) == 3 and clean_u.isalpha():
        return clean_u

    return u_val


def create_resilient_session() -> requests.Session:
    """
    Crea una sesión HTTP configurada con política de reintentos exponencial y headers modernos.
    - total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    - User-Agent moderno y Accept: application/json
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _normalize_date_to_espn(d_str: str) -> str:
    """Convierte YYYY-MM-DD, DD/MM/YYYY o YYYYMMDD a formato continuo YYYYMMDD."""
    d_clean = str(d_str).strip()
    if not d_clean:
        return ""
    if len(d_clean) == 8 and d_clean.isdigit():
        return d_clean
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(d_clean, fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            continue
    digits = re.sub(r"\D", "", d_clean)
    if len(digits) == 8:
        return digits
    return d_clean


# ---------------------------------------------------------------------------
# Proveedor de Datos de Equipo (TeamDataProvider)
# ---------------------------------------------------------------------------
class NBAApiTeamDataProvider(TeamDataProvider):
    """
    Obtiene datos de equipos en tiempo real desde stats.nba.com.
    Importa nba_api de forma lazy para no romper el inicio si no está instalado.
    """

    def fetch(self, team_id: int, abbreviation: str, season: str) -> Optional[Team]:
        try:
            return self._fetch_internal(team_id, abbreviation, season)
        except ImportError:
            logger.error("nba_api no está instalado. Ejecuta: pip install nba_api")
            return None
        except Exception as exc:
            logger.error("Error obteniendo datos para %s: %s", abbreviation, exc)
            return None

    def _fetch_internal(self, team_id: int, abbreviation: str, season: str) -> Optional[Team]:
        from nba_api.stats.endpoints import teamgamelogs, playergamelogs

        logger.info("Solicitando logs del equipo %s a NBA API...", abbreviation)

        # --- Estadísticas del equipo ---
        logs_df = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]

        if logs_df.empty:
            logger.warning("Sin datos de logs para el equipo %s", team_id)
            return None

        stats = TeamStats.from_dict({
            col: float(logs_df[col].mean())
            for col in _STAT_COLS
            if col in logs_df.columns
        })

        # --- Top jugadores por puntos y minutos ---
        player_df = playergamelogs.PlayerGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]

        top_players = self._extract_top_players(player_df)

        logger.info("Datos obtenidos exitosamente para %s", abbreviation)

        return Team(
            team_id=team_id,
            full_name=abbreviation,
            abbreviation=abbreviation,
            season=season,
            stats=stats,
            top_players=top_players,
            fetched_at=datetime.utcnow(),
            source="nba_api",
        )

    @staticmethod
    def _extract_top_players(player_df) -> List[PlayerStat]:
        if player_df.empty:
            return []
        grouped = player_df.groupby("PLAYER_NAME").agg(
            avg_pts=("PTS", "mean"),
            avg_min=("MIN", "mean"),
        ).reset_index()
        top = grouped.nlargest(5, "avg_pts")
        return [
            PlayerStat(
                name=row["PLAYER_NAME"],
                avg_points=round(float(row["avg_pts"]), 1),
                avg_minutes=round(float(row["avg_min"]), 1),
            )
            for _, row in top.iterrows()
        ]


# ---------------------------------------------------------------------------
# Fallback Cascade: Tier 1 (ESPN Public API)
# ---------------------------------------------------------------------------
def _fetch_scoreboard_espn(
    dates_to_check: List[str],
    session: Optional[requests.Session] = None,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Tier 1 (Primaria): Consulta la API Pública de ESPN.
    Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={YYYYMMDD}
    Condición de cierre: event['competitions'][0]['status']['type']['name'] == 'STATUS_FINAL'
    """
    sess = session or create_resilient_session()
    espn_results: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for d_str in dates_to_check:
        espn_date = _normalize_date_to_espn(d_str)
        if not espn_date:
            continue

        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_date}"
        try:
            resp = sess.get(url, timeout=DEFAULT_TIMEOUT)
            if resp.status_code != 200:
                logger.debug("ESPN scoreboard error HTTP %s for date %s", resp.status_code, espn_date)
                continue

            data = resp.json()
            events = data.get("events", [])
            for event in events:
                comps = event.get("competitions", [])
                if not comps:
                    continue
                comp = comps[0]
                status_node = comp.get("status", {})
                status_type = status_node.get("type", {})
                status_name = str(status_type.get("name", "")).strip().upper()
                completed = bool(status_type.get("completed", False))

                # Condición de cierre: STATUS_FINAL o completed == True
                if status_name != "STATUS_FINAL" and not completed:
                    continue

                home_abbr, away_abbr = None, None
                home_pts, away_pts = None, None

                for competitor in comp.get("competitors", []):
                    h_a = competitor.get("homeAway")
                    team_info = competitor.get("team", {})
                    team_raw = (
                        team_info.get("abbreviation")
                        or team_info.get("shortDisplayName")
                        or team_info.get("displayName")
                        or ""
                    )
                    team_code = standardize_team_code(team_raw)
                    score_raw = competitor.get("score")
                    try:
                        score_val = int(score_raw)
                    except (ValueError, TypeError):
                        score_val = 0

                    if h_a == "home":
                        home_abbr = team_code
                        home_pts = score_val
                    elif h_a == "away":
                        away_abbr = team_code
                        away_pts = score_val

                if home_abbr and away_abbr and home_pts is not None and away_pts is not None:
                    espn_results[(home_abbr, away_abbr)] = (home_pts, away_pts)

        except Exception as exc:
            logger.debug("ESPN scoreboard request failed for date %s: %s", espn_date, exc)
            continue

    return espn_results


# ---------------------------------------------------------------------------
# Fallback Cascade: Tier 2 (NBA Live CDN Edge)
# ---------------------------------------------------------------------------
def _fetch_scoreboard_nba_cdn(
    session: Optional[requests.Session] = None,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Tier 2 (Secundaria): Consulta NBA Live CDN Edge.
    Endpoint: https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_v2.json
    Condición de cierre: game_node['gameStatus'] == 3
    """
    sess = session or create_resilient_session()
    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_v2.json"
    cdn_results: Dict[Tuple[str, str], Tuple[int, int]] = {}

    try:
        resp = sess.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            logger.debug("NBA CDN scoreboard returned HTTP %s", resp.status_code)
            return {}

        data = resp.json()
        scoreboard = data.get("scoreboard", {})
        games = scoreboard.get("games", [])
        for g in games:
            status_id = g.get("gameStatus")
            status_text = str(g.get("gameStatusText") or "").strip().lower()

            is_final = False
            try:
                if status_id is not None and int(status_id) == 3:
                    is_final = True
            except (ValueError, TypeError):
                pass

            if "final" in status_text:
                is_final = True

            if not is_final:
                continue

            h_team = g.get("homeTeam", {})
            a_team = g.get("awayTeam", {})
            h_code = standardize_team_code(h_team.get("teamTricode") or "")
            a_code = standardize_team_code(a_team.get("teamTricode") or "")

            try:
                h_pts = int(h_team.get("score", 0))
                a_pts = int(a_team.get("score", 0))
            except (ValueError, TypeError):
                continue

            if h_code and a_code:
                cdn_results[(h_code, a_code)] = (h_pts, a_pts)

    except Exception as exc:
        logger.debug("NBA CDN scoreboard request failed: %s", exc)

    return cdn_results


# ---------------------------------------------------------------------------
# Fallback Cascade: Tier 3 (The Odds API / ScoreboardV2)
# ---------------------------------------------------------------------------
def _fetch_scoreboard_odds_api(
    session: Optional[requests.Session] = None,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Tier 3 (Contingencia A): Consulta endpoint de scores de The Odds API si hay API key configurada.
    """
    try:
        from infrastructure.config import config
        api_key = getattr(config, "odds_api_key", None)
        if not api_key:
            return {}

        sess = session or create_resilient_session()
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/scores/?daysFrom=3&apiKey={api_key}"
        resp = sess.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        odds_results: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for game in data:
            if not game.get("completed"):
                continue
            scores = game.get("scores")
            if not scores or len(scores) < 2:
                continue

            home_team_raw = game.get("home_team")
            away_team_raw = game.get("away_team")
            home_code = standardize_team_code(home_team_raw)
            away_code = standardize_team_code(away_team_raw)

            home_pts, away_pts = None, None
            for sc in scores:
                sc_name = sc.get("name")
                sc_val = sc.get("score")
                try:
                    sc_int = int(sc_val)
                except (ValueError, TypeError):
                    continue

                if sc_name == home_team_raw or standardize_team_code(sc_name) == home_code:
                    home_pts = sc_int
                elif sc_name == away_team_raw or standardize_team_code(sc_name) == away_code:
                    away_pts = sc_int

            if home_code and away_code and home_pts is not None and away_pts is not None:
                odds_results[(home_code, away_code)] = (home_pts, away_pts)

        return odds_results
    except Exception as exc:
        logger.debug("The Odds API scores fetch failed: %s", exc)
        return {}


def _fetch_scoreboard_nba_api_fallback(
    dates_to_check: List[str],
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Tier 3 (Contingencia B): Fallback encapsulado a stats.nba.com (ScoreboardV2) con manejo seguro de errores.
    """
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except ImportError:
        return {}

    nba_api_results: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for d_str in dates_to_check:
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=d_str)
            dfs = sb.get_data_frames()
            if len(dfs) < 2:
                continue
            header_df = dfs[0]
            lines_df = dfs[1]

            if header_df.empty or lines_df.empty:
                continue

            for _, h_row in header_df.iterrows():
                gid = str(h_row.get("GAME_ID"))
                status_id = h_row.get("GAME_STATUS_ID")
                status_text = str(h_row.get("GAME_STATUS_TEXT") or "").strip().lower()

                is_final = False
                try:
                    if status_id is not None and int(status_id) == 3:
                        is_final = True
                except (ValueError, TypeError):
                    pass

                if "final" in status_text:
                    is_final = True

                if not is_final:
                    continue

                home_id = h_row.get("HOME_TEAM_ID")
                visitor_id = h_row.get("VISITOR_TEAM_ID")

                h_line = lines_df[(lines_df["GAME_ID"] == gid) & (lines_df["TEAM_ID"] == home_id)]
                v_line = lines_df[(lines_df["GAME_ID"] == gid) & (lines_df["TEAM_ID"] == visitor_id)]

                if not h_line.empty and not v_line.empty:
                    h_pts = int(h_line.iloc[0].get("PTS") or 0)
                    v_pts = int(v_line.iloc[0].get("PTS") or 0)
                    h_abbr = standardize_team_code(h_line.iloc[0].get("TEAM_ABBREVIATION") or "")
                    v_abbr = standardize_team_code(v_line.iloc[0].get("TEAM_ABBREVIATION") or "")

                    if h_abbr and v_abbr:
                        nba_api_results[(h_abbr, v_abbr)] = (h_pts, v_pts)
        except Exception as exc:
            logger.debug("ScoreboardV2 fallback exception for date %s: %s", d_str, exc)
            continue

    return nba_api_results


# ---------------------------------------------------------------------------
# Función Orquestadora Principal (Arquitectura Fallback Cascade)
# ---------------------------------------------------------------------------
def fetch_scoreboard_results(
    game_date: Optional[Union[str, List[str]]] = None,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Consulta los marcadores finales oficiales de la NBA mediante una Arquitectura
    de Liquidación en Cascada (Fallback Cascade) resiliente:
      - Tier 1 (Primaria): API Pública de ESPN.
      - Tier 2 (Secundaria): NBA Live CDN Edge.
      - Tier 3 (Contingencia): The Odds API (/scores) y ScoreboardV2.

    Retorna diccionario canónico: {(home_abbr, away_abbr): (home_pts, away_pts)}.
    Solo incluye encuentros con estado oficial final.
    """
    try:
        dates_to_check: List[str] = []
        if isinstance(game_date, list):
            for gd in game_date:
                if gd and str(gd).strip():
                    dates_to_check.append(str(gd).strip())
        elif game_date:
            dates_to_check.append(str(game_date).strip())
        else:
            try:
                from infrastructure.reporting.bet_tracker_excel import get_local_now
                today_dt = get_local_now().date()
            except Exception:
                today_dt = datetime.now(timezone.utc).date()
            for d in range(4):
                dates_to_check.append((today_dt - timedelta(days=d)).strftime("%Y-%m-%d"))

        # Deduplicar preservando orden
        dates_to_check = list(dict.fromkeys(dates_to_check))

        session = create_resilient_session()
        results: Dict[Tuple[str, str], Tuple[int, int]] = {}

        # -------------------------------------------------------------------
        # Tier 1: API Pública de ESPN (Primaria)
        # -------------------------------------------------------------------
        try:
            tier1_results = _fetch_scoreboard_espn(dates_to_check, session=session)
            if tier1_results:
                logger.info("Marcadores finales obtenidos exitosamente vía ESPN (Tier 1): %d partidos", len(tier1_results))
                results.update(tier1_results)
        except Exception as exc:
            logger.debug("Tier 1 (ESPN) cascade trigger: %s", exc)

        # -------------------------------------------------------------------
        # Tier 2: NBA Live CDN Edge (Secundaria)
        # -------------------------------------------------------------------
        # Si Tier 1 no obtuvo resultados, o para complementar juegos en curso hoy
        if not results:
            try:
                tier2_results = _fetch_scoreboard_nba_cdn(session=session)
                if tier2_results:
                    logger.info("Marcadores finales obtenidos vía NBA CDN Edge (Tier 2): %d partidos", len(tier2_results))
                    for k, v in tier2_results.items():
                        if k not in results:
                            results[k] = v
            except Exception as exc:
                logger.debug("Tier 2 (NBA CDN) cascade trigger: %s", exc)

        # -------------------------------------------------------------------
        # Tier 3: The Odds API & ScoreboardV2 (Contingencia)
        # -------------------------------------------------------------------
        if not results:
            # Contingencia A: The Odds API scores
            try:
                tier3_odds = _fetch_scoreboard_odds_api(session=session)
                if tier3_odds:
                    logger.info("Marcadores finales obtenidos vía The Odds API (Tier 3): %d partidos", len(tier3_odds))
                    results.update(tier3_odds)
            except Exception as exc:
                logger.debug("Tier 3 (The Odds API) cascade trigger: %s", exc)

            # Contingencia B: ScoreboardV2 nba_api
            if not results:
                try:
                    tier3_sb = _fetch_scoreboard_nba_api_fallback(dates_to_check)
                    if tier3_sb:
                        logger.info("Marcadores finales obtenidos vía ScoreboardV2 (Tier 3): %d partidos", len(tier3_sb))
                        results.update(tier3_sb)
                except Exception as exc:
                    logger.debug("Tier 3 (ScoreboardV2) cascade trigger: %s", exc)

        return results

    except Exception as exc:
        logger.warning("scoreboard-cascade-fetch-error: %s", exc)
        return {}


__all__ = [
    "NBAApiTeamDataProvider",
    "fetch_scoreboard_results",
    "standardize_team_code",
    "TEAM_CANONICAL_MAP",
    "create_resilient_session",
    "DEFAULT_TIMEOUT",
]

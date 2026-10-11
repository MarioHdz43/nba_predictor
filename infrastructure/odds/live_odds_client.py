# -*- coding: utf-8 -*-
"""
infrastructure/odds/live_odds_client.py
---------------------------------------
Cliente institucional para extracción de cuotas en vivo de la NBA vía The Odds API.
Realiza consultas seguras con tolerancia a fallos, mapeo canónico de franquicias,
resolución de bookmakers institucionales (Pinnacle/DraftKings) y caché temporal.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import zoneinfo
except ImportError:
    zoneinfo = None

import requests
import structlog

from infrastructure.config import config

logger = structlog.get_logger()

# Mapeo canónico exhaustivo de las 30 franquicias de la NBA (claves -> abreviatura canónica)
NBA_TEAM_ALIASES: Dict[str, str] = {
    # Atlanta Hawks
    "atlanta hawks": "ATL", "atlanta": "ATL", "hawks": "ATL", "atl": "ATL",
    # Boston Celtics
    "boston celtics": "BOS", "boston": "BOS", "celtics": "BOS", "bos": "BOS",
    # Brooklyn Nets
    "brooklyn nets": "BKN", "brooklyn": "BKN", "nets": "BKN", "bkn": "BKN",
    # Charlotte Hornets
    "charlotte hornets": "CHA", "charlotte": "CHA", "hornets": "CHA", "cha": "CHA",
    # Chicago Bulls
    "chicago bulls": "CHI", "chicago": "CHI", "bulls": "CHI", "chi": "CHI",
    # Cleveland Cavaliers
    "cleveland cavaliers": "CLE", "cleveland": "CLE", "cavaliers": "CLE", "cavs": "CLE", "cle": "CLE",
    # Dallas Mavericks
    "dallas mavericks": "DAL", "dallas": "DAL", "mavericks": "DAL", "mavs": "DAL", "dal": "DAL",
    # Denver Nuggets
    "denver nuggets": "DEN", "denver": "DEN", "nuggets": "DEN", "den": "DEN",
    # Detroit Pistons
    "detroit pistons": "DET", "detroit": "DET", "pistons": "DET", "det": "DET",
    # Golden State Warriors
    "golden state warriors": "GSW", "golden state": "GSW", "warriors": "GSW", "gsw": "GSW",
    # Houston Rockets
    "houston rockets": "HOU", "houston": "HOU", "rockets": "HOU", "hou": "HOU",
    # Indiana Pacers
    "indiana pacers": "IND", "indiana": "IND", "pacers": "IND", "ind": "IND",
    # LA Clippers / Los Angeles Clippers
    "los angeles clippers": "LAC", "la clippers": "LAC", "clippers": "LAC", "lac": "LAC",
    # Los Angeles Lakers / LA Lakers
    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL", "lal": "LAL",
    # Memphis Grizzlies
    "memphis grizzlies": "MEM", "memphis": "MEM", "grizzlies": "MEM", "mem": "MEM",
    # Miami Heat
    "miami heat": "MIA", "miami": "MIA", "heat": "MIA", "mia": "MIA",
    # Milwaukee Bucks
    "milwaukee bucks": "MIL", "milwaukee": "MIL", "bucks": "MIL", "mil": "MIL",
    # Minnesota Timberwolves
    "minnesota timberwolves": "MIN", "minnesota": "MIN", "timberwolves": "MIN", "wolves": "MIN", "min": "MIN",
    # New Orleans Pelicans
    "new orleans pelicans": "NOP", "new orleans": "NOP", "pelicans": "NOP", "nop": "NOP",
    # New York Knicks
    "new york knicks": "NYK", "new york": "NYK", "knicks": "NYK", "nyk": "NYK",
    # Oklahoma City Thunder
    "oklahoma city thunder": "OKC", "oklahoma city": "OKC", "thunder": "OKC", "okc": "OKC",
    # Orlando Magic
    "orlando magic": "ORL", "orlando": "ORL", "magic": "ORL", "orl": "ORL",
    # Philadelphia 76ers
    "philadelphia 76ers": "PHI", "philadelphia": "PHI", "76ers": "PHI", "sixers": "PHI", "phi": "PHI",
    # Phoenix Suns
    "phoenix suns": "PHX", "phoenix": "PHX", "suns": "PHX", "phx": "PHX",
    # Portland Trail Blazers
    "portland trail blazers": "POR", "portland": "POR", "trail blazers": "POR", "blazers": "POR", "por": "POR",
    # Sacramento Kings
    "sacramento kings": "SAC", "sacramento": "SAC", "kings": "SAC", "sac": "SAC",
    # San Antonio Spurs
    "san antonio spurs": "SAS", "san antonio": "SAS", "spurs": "SAS", "sas": "SAS",
    # Toronto Raptors
    "toronto raptors": "TOR", "toronto": "TOR", "raptors": "TOR", "tor": "TOR",
    # Utah Jazz
    "utah jazz": "UTA", "utah": "UTA", "jazz": "UTA", "uta": "UTA",
    # Washington Wizards
    "washington wizards": "WAS", "washington": "WAS", "wizards": "WAS", "was": "WAS",
}


def canonicalize_team(team_str: Any) -> Optional[str]:
    """Convierte cualquier denominación de equipo NBA en su abreviatura oficial de 3 letras."""
    if not team_str:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", str(team_str).strip().lower())
    cleaned = " ".join(cleaned.split())
    return NBA_TEAM_ALIASES.get(cleaned)


def get_user_timezone() -> Any:
    """Obtiene la zona horaria de referencia (America/Bogota, UTC-5 o zona local del sistema)."""
    if zoneinfo is not None:
        try:
            return zoneinfo.ZoneInfo("America/Bogota")
        except Exception:
            pass
    sys_tz = datetime.now().astimezone().tzinfo
    if sys_tz is not None:
        return sys_tz
    return timezone(timedelta(hours=-5))


def parse_commence_time(
    commence_time_str: Optional[str],
    target_timezone: Optional[Any] = None,
    reference_now_utc: Optional[datetime] = None,
) -> Tuple[datetime, datetime, str, bool, str, str]:
    """
    Parsea commence_time en UTC (ISO 8601) y lo convierte a la hora local del usuario.

    Retorna:
    - dt_utc: datetime en UTC
    - dt_local: datetime en zona horaria local
    - local_time_str: Cadena formateada (ej: 'Hoy 06:30 PM', 'Mañana 07:00 PM', '20/10 02:00 PM')
    - is_pregame: True si el partido aún no ha comenzado (dt_utc > now_utc)
    - status_str: '[DISPONIBLE - HH:MM PM]' si is_pregame else '[EN JUGADA / CERRADO]'
    - status_tag: '[DISPONIBLE]' si is_pregame else '[EN JUGADA / CERRADO]'
    """
    tz_local = target_timezone or get_user_timezone()
    now_utc = reference_now_utc or datetime.now(timezone.utc)

    if not commence_time_str:
        now_loc = now_utc.astimezone(tz_local)
        return now_utc, now_loc, "Sin fecha", False, "[EN JUGADA / CERRADO]", "[EN JUGADA / CERRADO]"

    try:
        cleaned_iso = str(commence_time_str).strip()
        if cleaned_iso.endswith("Z"):
            cleaned_iso = cleaned_iso[:-1] + "+00:00"
        dt_utc = datetime.fromisoformat(cleaned_iso)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    except Exception:
        now_loc = now_utc.astimezone(tz_local)
        return now_utc, now_loc, "Fecha inválida", False, "[EN JUGADA / CERRADO]", "[EN JUGADA / CERRADO]"

    dt_local = dt_utc.astimezone(tz_local)
    now_local = now_utc.astimezone(tz_local)

    is_pregame = dt_utc > now_utc

    # Formateo relativo de fecha local
    date_diff = (dt_local.date() - now_local.date()).days
    if date_diff == 0:
        day_label = "Hoy"
    elif date_diff == 1:
        day_label = "Mañana"
    elif date_diff == -1:
        day_label = "Ayer"
    else:
        day_label = dt_local.strftime("%d/%m")

    time_part = dt_local.strftime("%I:%M %p")
    local_time_str = f"{day_label} {time_part}"

    if is_pregame:
        status_str = f"[DISPONIBLE - {time_part}]"
        status_tag = "[DISPONIBLE]"
    else:
        status_str = "[EN JUGADA / CERRADO]"
        status_tag = "[EN JUGADA / CERRADO]"

    return dt_utc, dt_local, local_time_str, is_pregame, status_str, status_tag


class LiveOddsClient:
    """
    Cliente de cuotas en vivo para la NBA conectándose a The Odds API.
    """

    BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    PREFERRED_BOOKMAKERS = [
        "pinnacle",
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "bovada",
        "betonlineag",
        "williamhill_us",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 8,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._api_key = api_key or getattr(config, "odds_api_key", None)
        self._timeout = timeout
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache_data: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_requests_remaining: Optional[str] = None
        self._last_requests_used: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key is not None:
            return self._api_key
        return getattr(config, "odds_api_key", None)

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self._api_key = value

    @api_key.deleter
    def api_key(self) -> None:
        self._api_key = None

    @property
    def requests_remaining(self) -> Optional[str]:
        """Consultas restantes de la cuota mensual reportadas por The Odds API."""
        return self._last_requests_remaining

    @property
    def requests_used(self) -> Optional[str]:
        """Consultas utilizadas reportadas por The Odds API."""
        return self._last_requests_used

    def clear_cache(self) -> None:
        """Limpia el caché de respuestas en memoria."""
        self._cache_data = None
        self._cache_timestamp = None

    def check_connection(self) -> Dict[str, Any]:
        """
        Verifica la conectividad directa con The Odds API y valida la API Key.
        Endpoint: v4/sports/basketball_nba/odds/?regions=us,eu&markets=h2h
        Retorna información de estado HTTP, cuota restante y número de partidos.
        """
        key = self.api_key
        if not key:
            return {
                "status_code": 0,
                "valid_key": False,
                "requests_remaining": None,
                "requests_used": None,
                "games_count": 0,
                "message": "No se ha configurado ninguna API Key en config.yaml o variable de entorno.",
            }

        try:
            params = {
                "apiKey": key,
                "regions": "us,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            }
            response = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            status_code = response.status_code
            rem = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            self._last_requests_remaining = rem
            self._last_requests_used = used

            if status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else 0
                return {
                    "status_code": 200,
                    "valid_key": True,
                    "requests_remaining": rem,
                    "requests_used": used,
                    "games_count": count,
                    "message": "Conexión exitosa y API Key válida.",
                }
            else:
                return {
                    "status_code": status_code,
                    "valid_key": False,
                    "requests_remaining": rem,
                    "requests_used": used,
                    "games_count": 0,
                    "message": f"Respuesta no exitosa ({status_code}): {response.text[:200]}",
                }
        except Exception as exc:
            return {
                "status_code": 0,
                "valid_key": False,
                "requests_remaining": None,
                "requests_used": None,
                "games_count": 0,
                "message": f"Error de conexión: {str(exc)}",
            }

    def _fetch_active_games(self) -> List[Dict[str, Any]]:
        """
        Consulta los partidos activos y cuotas moneyline (h2h) en The Odds API.
        Usa caché en memoria si la consulta previa ocurrió dentro de cache_ttl.
        """
        now = datetime.utcnow()
        if (
            self._cache_data is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp) < self._cache_ttl
        ):
            return self._cache_data

        key = self.api_key
        if not key:
            logger.debug("odds-api-key-missing", msg="No ODDS_API_KEY configurada.")
            return []

        try:
            params = {
                "apiKey": key,
                "regions": "us,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            }
            response = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            self._last_requests_remaining = response.headers.get("x-requests-remaining")
            self._last_requests_used = response.headers.get("x-requests-used")

            if self._last_requests_remaining is not None:
                logger.info(
                    "odds-api-quota",
                    remaining=self._last_requests_remaining,
                    used=self._last_requests_used,
                )

            if response.status_code != 200:
                logger.warning(
                    "odds-api-request-failed",
                    status_code=response.status_code,
                    body=response.text[:200],
                )
                return []

            data = response.json()
            if isinstance(data, list):
                self._cache_data = data
                self._cache_timestamp = now
                logger.info("odds-api-games-fetched", count=len(data))
                return data

            return []
        except Exception as exc:
            logger.warning("odds-api-connection-error", error=str(exc))
            return []

    def _extract_odds_from_game(
        self,
        game: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extrae las cuotas h2h para home_team y away_team del diccionario del partido.
        Retorna (home_odds, away_odds, bookmaker_title).
        """
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            return None, None, None

        home_abbr = canonicalize_team(game.get("home_team"))
        away_abbr = canonicalize_team(game.get("away_team"))

        bk_by_key = {b.get("key", "").lower(): b for b in bookmakers}

        selected_bk = None
        for pref in self.PREFERRED_BOOKMAKERS:
            if pref in bk_by_key:
                selected_bk = bk_by_key[pref]
                break

        if selected_bk is None:
            selected_bk = bookmakers[0]

        markets = selected_bk.get("markets", [])
        for market in markets:
            if market.get("key") == "h2h":
                outcomes = market.get("outcomes", [])
                h_price: Optional[float] = None
                a_price: Optional[float] = None

                for out in outcomes:
                    out_team = canonicalize_team(out.get("name"))
                    price = float(out.get("price", 0.0))
                    if out_team == home_abbr:
                        h_price = price
                    elif out_team == away_abbr:
                        a_price = price

                if h_price and a_price and h_price > 1.0 and a_price > 1.0:
                    bk_title = selected_bk.get("title", "Consenso")
                    return round(h_price, 3), round(a_price, 3), bk_title

        return None, None, None

    def list_available_games(
        self,
        target_timezone: Optional[Any] = None,
        only_pregame: bool = False,
        reference_now_utc: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lista los partidos del mercado estructurados con horario local, categoría de fecha y estado.
        Retorna lista de diccionarios con:
        - home_team: str
        - away_team: str
        - home_abbr: str
        - away_abbr: str
        - home_odds: Optional[float]
        - away_odds: Optional[float]
        - bookmaker: Optional[str]
        - local_time_str: str (ej: 'Hoy 06:30 PM')
        - is_pregame: bool
        - status_str: str
        - status_tag: str
        - date_category: str ('HOY', 'MAÑANA', 'PRÓXIMOS', 'AYER', 'PASADOS')
        - game_id: str
        - commence_time_utc: datetime
        - commence_time_local: datetime
        """
        games = self._fetch_active_games()
        if not games:
            return []

        tz_local = target_timezone or get_user_timezone()
        now_utc = reference_now_utc or datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz_local)

        results: List[Dict[str, Any]] = []
        for g in games:
            h_team = g.get("home_team", "")
            a_team = g.get("away_team", "")
            h_abbr = canonicalize_team(h_team) or h_team[:3].upper()
            a_abbr = canonicalize_team(a_team) or a_team[:3].upper()

            dt_utc, dt_loc, loc_str, is_pre, st_str, st_tag = parse_commence_time(
                g.get("commence_time"),
                target_timezone=tz_local,
                reference_now_utc=now_utc,
            )

            if only_pregame and not is_pre:
                continue

            h_odds, a_odds, bk_name = self._extract_odds_from_game(g)

            # Categorización por fecha relativa local
            commence_raw = g.get("commence_time")
            if not commence_raw or loc_str in ("Sin fecha", "Fecha inválida"):
                date_cat = "DESCONOCIDO"
            else:
                date_diff = (dt_loc.date() - now_local.date()).days
                if date_diff == 0:
                    date_cat = "HOY"
                elif date_diff == 1:
                    date_cat = "MAÑANA"
                elif date_diff > 1:
                    date_cat = "PRÓXIMOS"
                elif date_diff == -1:
                    date_cat = "AYER"
                else:
                    date_cat = "PASADOS"

            results.append({
                "game_id": str(g.get("id") or ""),
                "home_team": h_team,
                "away_team": a_team,
                "home_abbr": h_abbr,
                "away_abbr": a_abbr,
                "home_odds": h_odds,
                "away_odds": a_odds,
                "bookmaker": bk_name,
                "commence_time_utc": dt_utc,
                "commence_time_local": dt_loc,
                "local_time_str": loc_str,
                "is_pregame": is_pre,
                "status_str": st_str,
                "status_tag": st_tag,
                "date_category": date_cat,
            })

        results.sort(key=lambda x: x["commence_time_utc"])
        return results

    def get_games_by_filter(
        self,
        filter_type: str = "today",
        target_timezone: Optional[Any] = None,
        only_pregame: bool = False,
        reference_now_utc: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna partidos filtrados por categoría de fecha relativa o fecha específica.

        filter_type:
        - 'today' / 'hoy': Partidos con categoría 'HOY'
        - 'tomorrow' / 'manana' / 'mañana': Partidos con categoría 'MAÑANA'
        - 'upcoming' / 'proximos' / 'próximos': Partidos futuros con categoría 'PRÓXIMOS'
        - 'all' / 'todos': Todos los partidos programados
        - Formato 'YYYY-MM-DD' o 'DD/MM': Coincidencia exacta de fecha local
        """
        all_games = self.list_available_games(
            target_timezone=target_timezone,
            only_pregame=only_pregame,
            reference_now_utc=reference_now_utc,
        )
        if not all_games:
            return []

        norm = str(filter_type).strip().lower()

        if norm in ("all", "todos", "*"):
            return all_games

        if norm in ("today", "hoy"):
            return [g for g in all_games if g.get("date_category") == "HOY"]

        if norm in ("tomorrow", "manana", "mañana"):
            return [g for g in all_games if g.get("date_category") == "MAÑANA"]

        if norm in ("upcoming", "proximos", "próximos"):
            return [g for g in all_games if g.get("date_category") == "PRÓXIMOS"]

        if norm in ("yesterday", "ayer"):
            return [g for g in all_games if g.get("date_category") == "AYER"]

        # Coincidencia por fecha local específica
        matched: List[Dict[str, Any]] = []
        for g in all_games:
            dt_loc: Optional[datetime] = g.get("commence_time_local")
            if dt_loc is not None:
                if (
                    dt_loc.strftime("%Y-%m-%d") == norm
                    or dt_loc.strftime("%d/%m") == norm
                    or dt_loc.strftime("%d/%m/%Y") == norm
                ):
                    matched.append(g)
        return matched

    def group_games_by_date(
        self,
        target_timezone: Optional[Any] = None,
        only_pregame: bool = False,
        reference_now_utc: Optional[datetime] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Agrupa los partidos por categoría de fecha local:
        {
            'HOY': [...],
            'MAÑANA': [...],
            'PRÓXIMOS': [...]
        }
        """
        all_games = self.list_available_games(
            target_timezone=target_timezone,
            only_pregame=only_pregame,
            reference_now_utc=reference_now_utc,
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {
            "HOY": [],
            "MAÑANA": [],
            "PRÓXIMOS": [],
        }
        for g in all_games:
            cat = g.get("date_category", "PRÓXIMOS")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(g)
        return grouped

    def find_market_match(
        self,
        team_a: str,
        team_b: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Busca un partido en el mercado activo entre dos equipos, detectando
        si la localía coincide de forma directa o si fue invertida por el usuario.

        Retorna un dict con:
        - official_home: Nombre completo oficial del equipo local
        - official_away: Nombre completo oficial del equipo visitante
        - official_home_abbr: Abreviatura oficial del local (3 letras)
        - official_away_abbr: Abreviatura oficial del visitante (3 letras)
        - home_odds: Cuota del local oficial
        - away_odds: Cuota del visitante oficial
        - bookmaker: Casa de apuestas seleccionada
        - is_inverted: True si team_a fue ingresado como local pero es visitante oficial
        - local_time_str: Texto formateado de la hora local
        - is_pregame: True si aún no ha iniciado
        - status_str: '[DISPONIBLE - HH:MM PM]' o '[EN JUGADA / CERRADO]'
        - status_tag: '[DISPONIBLE]' o '[EN JUGADA / CERRADO]'
        - game_id: ID del evento en The Odds API
        - commence_time_utc: datetime UTC
        - commence_time_local: datetime local
        """
        abbr_a = canonicalize_team(team_a)
        abbr_b = canonicalize_team(team_b)
        if not abbr_a or not abbr_b:
            logger.debug("unknown-teams-for-odds", team_a=team_a, team_b=team_b)
            return None

        games = self._fetch_active_games()
        if not games:
            return None

        for g in games:
            g_home = canonicalize_team(g.get("home_team"))
            g_away = canonicalize_team(g.get("away_team"))

            # Coincidencia directa: team_a es local oficial, team_b es visitante oficial
            if g_home == abbr_a and g_away == abbr_b:
                h_odds, a_odds, bk = self._extract_odds_from_game(g)
                dt_utc, dt_loc, loc_str, is_pre, st_str, st_tag = parse_commence_time(g.get("commence_time"))
                return {
                    "game_id": str(g.get("id") or ""),
                    "official_home": g.get("home_team"),
                    "official_away": g.get("away_team"),
                    "official_home_abbr": g_home,
                    "official_away_abbr": g_away,
                    "home_odds": h_odds,
                    "away_odds": a_odds,
                    "bookmaker": bk,
                    "is_inverted": False,
                    "local_time_str": loc_str,
                    "is_pregame": is_pre,
                    "status_str": st_str,
                    "status_tag": st_tag,
                    "commence_time_utc": dt_utc,
                    "commence_time_local": dt_loc,
                }

            # Coincidencia invertida: team_b es local oficial, team_a es visitante oficial
            if g_home == abbr_b and g_away == abbr_a:
                h_odds, a_odds, bk = self._extract_odds_from_game(g)
                dt_utc, dt_loc, loc_str, is_pre, st_str, st_tag = parse_commence_time(g.get("commence_time"))
                return {
                    "game_id": str(g.get("id") or ""),
                    "official_home": g.get("home_team"),
                    "official_away": g.get("away_team"),
                    "official_home_abbr": g_home,
                    "official_away_abbr": g_away,
                    "home_odds": h_odds,
                    "away_odds": a_odds,
                    "bookmaker": bk,
                    "is_inverted": True,
                    "local_time_str": loc_str,
                    "is_pregame": is_pre,
                    "status_str": st_str,
                    "status_tag": st_tag,
                    "commence_time_utc": dt_utc,
                    "commence_time_local": dt_loc,
                }

        return None

    def get_game_odds(
        self,
        home_team: str,
        away_team: str,
    ) -> Optional[Tuple[float, float, str]]:
        """
        Busca las cuotas de un partido específico según los equipos local y visitante.
        Retorna (home_odds, away_odds, bookmaker_name) o None si no hay líneas activas.
        """
        match = self.find_market_match(home_team, away_team)
        if match is None:
            return None

        # Si el usuario pasó los equipos invertidos, devuelve las cuotas en el orden
        # correspondiente a los equipos pasados: (cuota de home_team, cuota de away_team)
        if match["is_inverted"]:
            h_odd = match.get("away_odds")  # home_team es el visitante oficial
            a_odd = match.get("home_odds")  # away_team es el local oficial
        else:
            h_odd = match.get("home_odds")
            a_odd = match.get("away_odds")

        if h_odd is not None and a_odd is not None and h_odd > 1.0 and a_odd > 1.0:
            return h_odd, a_odd, match.get("bookmaker") or "Consenso"

        return None


_default_client = LiveOddsClient()


def get_game_odds(
    home_team: str,
    away_team: str,
    api_key: Optional[str] = None,
) -> Optional[Tuple[float, float, str]]:
    """
    Función de acceso directo para consultar cuotas en vivo.
    """
    if api_key:
        client = LiveOddsClient(api_key=api_key)
        return client.get_game_odds(home_team, away_team)
    return _default_client.get_game_odds(home_team, away_team)


def list_available_games(
    target_timezone: Optional[Any] = None,
    only_pregame: bool = False,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Función de acceso directo para listar partidos disponibles en el mercado.
    """
    if api_key:
        client = LiveOddsClient(api_key=api_key)
        return client.list_available_games(target_timezone=target_timezone, only_pregame=only_pregame)
    return _default_client.list_available_games(target_timezone=target_timezone, only_pregame=only_pregame)


def find_market_match(
    team_a: str,
    team_b: str,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Función de acceso directo para buscar un partido en el mercado y verificar localía.
    """
    if api_key:
        client = LiveOddsClient(api_key=api_key)
        return client.find_market_match(team_a, team_b)
    return _default_client.find_market_match(team_a, team_b)


def get_games_by_filter(
    filter_type: str = "today",
    target_timezone: Optional[Any] = None,
    only_pregame: bool = False,
    api_key: Optional[str] = None,
    reference_now_utc: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Función de acceso directo para consultar partidos filtrados por categoría de fecha.
    """
    if api_key:
        client = LiveOddsClient(api_key=api_key)
        return client.get_games_by_filter(
            filter_type=filter_type,
            target_timezone=target_timezone,
            only_pregame=only_pregame,
            reference_now_utc=reference_now_utc,
        )
    return _default_client.get_games_by_filter(
        filter_type=filter_type,
        target_timezone=target_timezone,
        only_pregame=only_pregame,
        reference_now_utc=reference_now_utc,
    )


def group_games_by_date(
    target_timezone: Optional[Any] = None,
    only_pregame: bool = False,
    api_key: Optional[str] = None,
    reference_now_utc: Optional[datetime] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Función de acceso directo para agrupar partidos por fecha local (HOY, MAÑANA, PRÓXIMOS).
    """
    if api_key:
        client = LiveOddsClient(api_key=api_key)
        return client.group_games_by_date(
            target_timezone=target_timezone,
            only_pregame=only_pregame,
            reference_now_utc=reference_now_utc,
        )
    return _default_client.group_games_by_date(
        target_timezone=target_timezone,
        only_pregame=only_pregame,
        reference_now_utc=reference_now_utc,
    )


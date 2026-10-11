# -*- coding: utf-8 -*-
"""
scripts/fetch_historical_data.py
--------------------------------
Script ejecutable e idempotente para la ingesta masiva de partidos y estadísticas históricas
reales de la NBA desde stats.nba.com a través de nba_api.

Uso:
    python scripts/fetch_historical_data.py --seasons 2022-23 2023-24
    python scripts/fetch_historical_data.py --help
"""

from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Permitir ejecución directa del script agregando el directorio raíz al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import structlog
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog

from infrastructure.database import Database, init_db
from infrastructure.historical_repository import SQLiteHistoricalRepository

logger = structlog.get_logger()

# Soporte opcional para barra de progreso con tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

DEFAULT_SEASON_TYPES = [
    "Regular Season",
]


def parse_gamelog_dataframe(
    df: pd.DataFrame,
    season: str,
    season_type: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transforma el DataFrame retornado por LeagueGameLog en diccionarios para
    historical_games y historical_team_boxscores.

    Garantiza que cada partido tenga identificado su equipo local y visitante
    a partir de la columna MATCHUP ('vs.' = local, '@' = visitante).
    """
    if df is None or df.empty:
        return [], []

    # Normalizar nombres de columnas a mayúsculas
    df = df.copy()
    df.columns = [col.upper() for col in df.columns]

    games_dict: Dict[str, Dict[str, Any]] = {}
    boxscores_list: List[Dict[str, Any]] = []

    # Agrupar filas por partido (cada fila corresponde al boxscore de un equipo en un juego)
    grouped = df.groupby("GAME_ID")

    for raw_game_id, group in grouped:
        game_id = str(raw_game_id).zfill(10)

        # Detectar local y visitante
        # Si MATCHUP contiene '@', el equipo es visitante; si contiene 'vs' o 'vs.', es local
        home_row = None
        away_row = None

        for _, row in group.iterrows():
            matchup = str(row.get("MATCHUP", ""))
            is_home = 0 if "@" in matchup else 1

            if is_home == 1 and home_row is None:
                home_row = row
            elif is_home == 0 and away_row is None:
                away_row = row

        # Si el partido no tiene dos filas claramente separadas (caso infrecuente),
        # usamos el primer registro como local y el segundo como visitante si difieren de equipo
        if home_row is None or away_row is None:
            if len(group) >= 2:
                rows = [r for _, r in group.iterrows()]
                home_row = rows[0]
                away_row = rows[1]
            else:
                logger.warning("gamelog-partido-incompleto", game_id=game_id, rows_count=len(group))
                continue

        home_team_id = int(home_row["TEAM_ID"])
        away_team_id = int(away_row["TEAM_ID"])
        home_score = int(home_row["PTS"]) if pd.notnull(home_row["PTS"]) else 0
        away_score = int(away_row["PTS"]) if pd.notnull(away_row["PTS"]) else 0
        game_date = str(home_row.get("GAME_DATE", ""))

        games_dict[game_id] = {
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_score": home_score,
            "away_score": away_score,
            "season_type": season_type,
        }

        # Extraer boxscore por equipo
        for _, row in group.iterrows():
            matchup = str(row.get("MATCHUP", ""))
            is_home_flag = 0 if "@" in matchup else 1

            boxscores_list.append({
                "game_id": game_id,
                "team_id": int(row["TEAM_ID"]),
                "is_home": is_home_flag,
                "pts": int(row["PTS"]) if pd.notnull(row.get("PTS")) else None,
                "fgm": int(row["FGM"]) if pd.notnull(row.get("FGM")) else None,
                "fga": int(row["FGA"]) if pd.notnull(row.get("FGA")) else None,
                "fg3m": int(row["FG3M"]) if pd.notnull(row.get("FG3M")) else None,
                "fg3a": int(row["FG3A"]) if pd.notnull(row.get("FG3A")) else None,
                "ftm": int(row["FTM"]) if pd.notnull(row.get("FTM")) else None,
                "fta": int(row["FTA"]) if pd.notnull(row.get("FTA")) else None,
                "oreb": int(row["OREB"]) if pd.notnull(row.get("OREB")) else None,
                "dreb": int(row["DREB"]) if pd.notnull(row.get("DREB")) else None,
                "reb": int(row["REB"]) if pd.notnull(row.get("REB")) else None,
                "ast": int(row["AST"]) if pd.notnull(row.get("AST")) else None,
                "stl": int(row["STL"]) if pd.notnull(row.get("STL")) else None,
                "blk": int(row["BLK"]) if pd.notnull(row.get("BLK")) else None,
                "tov": int(row["TOV"]) if pd.notnull(row.get("TOV")) else None,
                "pf": int(row["PF"]) if pd.notnull(row.get("PF")) else None,
                "plus_minus": float(row["PLUS_MINUS"]) if pd.notnull(row.get("PLUS_MINUS")) else 0.0,
            })

    return list(games_dict.values()), boxscores_list


def fetch_season_gamelog_with_retry(
    season: str,
    season_type: str,
    delay_min: float = 0.6,
    delay_max: float = 1.2,
    max_retries: int = 4,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Descarga el registro de juegos de una temporada con rate limiting de cortesía
    y reintentos exponenciales.
    """
    # Pausa de cortesía para respetar la API
    time.sleep(random.uniform(delay_min, delay_max))

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("nba-api-solicitando", season=season, season_type=season_type, attempt=attempt)
            gamelog = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                timeout=timeout,
            )
            dfs = gamelog.get_data_frames()
            if dfs and len(dfs) > 0:
                logger.info(
                    "nba-api-descarga-exitosa",
                    season=season,
                    season_type=season_type,
                    rows=len(dfs[0]),
                )
                return dfs[0]
            return pd.DataFrame()
        except Exception as e:
            if attempt == max_retries:
                logger.error(
                    "nba-api-fallo-definitivo",
                    season=season,
                    season_type=season_type,
                    error=str(e),
                )
                raise
            backoff_seconds = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(
                "nba-api-reintentando",
                season=season,
                attempt=attempt,
                backoff_seconds=round(backoff_seconds, 2),
                error=str(e),
            )
            time.sleep(backoff_seconds)

    return pd.DataFrame()


def ingest_historical_data(
    seasons: List[str],
    season_types: List[str],
    repository: SQLiteHistoricalRepository,
    delay_min: float = 0.6,
    delay_max: float = 1.2,
    max_retries: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline principal de ingesta histórica:
    Itera por temporada, filtra partidos existentes y almacena atómicamente.
    """
    summary = {
        "seasons_processed": 0,
        "total_games_found": 0,
        "new_games_saved": 0,
        "new_boxscores_saved": 0,
    }

    total_tasks = len(seasons) * len(season_types)
    task_idx = 0

    print(f"\n=======================================================")
    print(f" NBA Historical Data Ingest Pipeline (LeagueGameLog)")
    print(f" Temporadas a procesar ({len(seasons)}): {', '.join(seasons)}")
    print(f" Tipos de temporada ({len(season_types)}): {', '.join(season_types)}")
    print(f" Modo Simulación (dry-run): {'ACTIVO' if dry_run else 'INACTIVO'}")
    print(f"=======================================================\n")

    iterator = tqdm(total=total_tasks, desc="Progreso total") if tqdm else None

    for season in seasons:
        # Recuperar IDs ya existentes para esta temporada (descarga incremental)
        existing_game_ids: Set[str] = repository.get_existing_game_ids([season])

        for season_type in season_types:
            task_idx += 1
            status_prefix = f"[{task_idx}/{total_tasks}] {season} ({season_type})"
            print(f"-> {status_prefix}: Descargando datos desde stats.nba.com...")

            df = fetch_season_gamelog_with_retry(
                season=season,
                season_type=season_type,
                delay_min=delay_min,
                delay_max=delay_max,
                max_retries=max_retries,
            )

            if df.empty:
                print(f"   [!] Sin datos disponibles.")
                if iterator:
                    iterator.update(1)
                continue

            games, boxscores = parse_gamelog_dataframe(df, season, season_type)
            summary["total_games_found"] += len(games)

            # Filtrar solo juegos que no existan previamente
            new_games = [g for g in games if g["game_id"] not in existing_game_ids]
            new_game_ids_set = {g["game_id"] for g in new_games}
            new_boxscores = [b for b in boxscores if b["game_id"] in new_game_ids_set]

            already_present = len(games) - len(new_games)
            print(
                f"   [OK] Juegos totales: {len(games)} | Ya en DB: {already_present} | Nuevos a guardar: {len(new_games)}"
            )

            if new_games and not dry_run:
                repository.save_games_batch(new_games, new_boxscores)
                # Actualizar set en memoria para evitar colisiones
                existing_game_ids.update(new_game_ids_set)

            summary["new_games_saved"] += len(new_games)
            summary["new_boxscores_saved"] += len(new_boxscores)

            if iterator:
                iterator.update(1)

        summary["seasons_processed"] += 1

    if iterator:
        iterator.close()

    print("\n-------------------------------------------------------")
    print(" RESUMEN DE LA INGESTA HISTÓRICA:")
    print(f" - Temporadas procesadas: {summary['seasons_processed']}")
    print(f" - Total partidos detectados: {summary['total_games_found']}")
    print(f" - Nuevos partidos guardados: {summary['new_games_saved']}")
    print(f" - Nuevos boxscores guardados: {summary['new_boxscores_saved']}")
    print(f" - Partidos totales actuales en DB: {repository.count_games()}")
    print(f" - Boxscores totales actuales en DB: {repository.count_boxscores()}")
    print("-------------------------------------------------------\n")

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Configura los argumentos por CLI."""
    parser = argparse.ArgumentParser(
        description="Descarga e ingesta de datos históricos oficiales de la NBA mediante nba_api."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Lista de temporadas a descargar en formato YYYY-YY (ej: 2022-23 2023-24).",
    )
    parser.add_argument(
        "--season-types",
        nargs="+",
        default=DEFAULT_SEASON_TYPES,
        help="Tipos de temporada a procesar (ej: 'Regular Season', 'Playoffs').",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Ruta alternativa para la base de datos SQLite.",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=0.6,
        help="Tiempo mínimo de espera en segundos entre llamadas (default: 0.6).",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=1.2,
        help="Tiempo máximo de espera en segundos entre llamadas (default: 1.2).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Máximo número de reintentos por petición ante fallas de red (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta la extracción y transformación sin guardar en la base de datos.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Inicializar base de datos
    db = Database(args.db_path) if args.db_path else Database()
    init_db()

    repo = SQLiteHistoricalRepository(db)

    ingest_historical_data(
        seasons=args.seasons,
        season_types=args.season_types,
        repository=repo,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

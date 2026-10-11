# -*- coding: utf-8 -*-
"""
main.py — Raíz de composición (Composition Root)
--------------------------------------------------
Este es el único lugar del sistema donde se instancian las implementaciones
concretas y se inyectan en los casos de uso. El resto del código solo conoce
las interfaces abstractas definidas en domain/repositories.py.

Para cambiar de CSV a SQLite, solo hace falta cambiar las dos líneas
donde se instancian CSVTeamRepository y CSVPredictionRepository.

Para cambiar el motor de predicción, inyecta una implementación diferente de
PredictionEngine (ej: XGBoostEngine, EnsembleEngine).
"""

import argparse

from application.use_cases import GetTeamDataUseCase, PredictGameUseCase
from domain.engines.heuristic_engine import HeuristicEngine
from infrastructure.config import config
from infrastructure.csv_repositories import CSVH2HRepository
from infrastructure.csv_repositories import CSVPredictionRepository, CSVTeamRepository
from infrastructure.database import Database, init_db
from infrastructure.logging_config import setup_logging
from infrastructure.nba_api_client import NBAApiTeamDataProvider
from infrastructure.sqlite_repositories import (
    SQLiteH2HRepository,
    SQLitePredictionRepository,
    SQLiteTeamRepository,
)
from presentation.cli import CLIController


def build_app(
    team_csv: str = None,
    history_csv: str = None,
    cache_ttl_hours: int = None,
    excel_path: str = None,
) -> CLIController:
    """
    Construye y cablea todas las dependencias.
    Retorna el controlador listo para ejecutar.

    Árbol de dependencias:
        CLIController
          └── PredictGameUseCase
                ├── GetTeamDataUseCase
                │     ├── CSVTeamRepository      (TeamRepository)
                │     └── NBAApiTeamDataProvider (TeamDataProvider)
                └── CSVPredictionRepository      (PredictionRepository)
                └── PredictionEngine (heuristic por defecto)
    """
    # Valores por defecto de config.yaml si no se proveen
    team_csv = team_csv or config.team_csv
    history_csv = history_csv or config.history_csv
    cache_ttl_hours = cache_ttl_hours or config.cache_ttl_hours
    repo_type = config.repository_type

    # Inicializar base de datos
    if repo_type == "sqlite":
        init_db()
        db = Database()
        team_repo = SQLiteTeamRepository(db)
        prediction_repo = SQLitePredictionRepository(db)
        h2h_repo = SQLiteH2HRepository(db)
    else:
        team_repo = CSVTeamRepository(path=team_csv)
        prediction_repo = CSVPredictionRepository(path=history_csv)
        h2h_repo = CSVH2HRepository()

    nba_provider = NBAApiTeamDataProvider()
    prediction_engine = HeuristicEngine()

    # Casos de uso
    get_team_data = GetTeamDataUseCase(
        repository=team_repo,
        provider=nba_provider,
        cache_ttl_hours=cache_ttl_hours,
    )
    predict_game = PredictGameUseCase(
        get_team_data=get_team_data,
        prediction_repo=prediction_repo,
        prediction_engine=prediction_engine,
        h2h_repo=h2h_repo,
    )

    # Presentación
    return CLIController(predict_game=predict_game, excel_path=excel_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de Predicción NBA — Arquitectura Limpia"
    )
    parser.add_argument("--home",       type=str, help="Equipo local (ID, abreviatura o nombre)")
    parser.add_argument("--away",       type=str, help="Equipo visitante")
    parser.add_argument("--season",     type=str, default="2024-25")
    parser.add_argument("--fresh",      action="store_true", help="Ignorar caché")
    parser.add_argument("--ttl",        type=int, default=2, help="Horas de validez del caché")
    parser.add_argument("--team-csv",   type=str, default="team_data.csv")
    parser.add_argument("--history-csv",type=str, default="prediction_history.csv")
    parser.add_argument("--excel-path", type=str, default=None, help="Ruta al archivo Excel de bitácora de apuestas")
    parser.add_argument("--settle-bets", action="store_true", help="Liquidar apuestas pendientes contra partidos finalizados")
    parser.add_argument("--settle-manual", action="store_true", help="Liquidar manualmente un partido pendiente en el Excel")
    parser.add_argument("--db-path",    type=str, default=None, help="Ruta a la base de datos SQLite")
    parser.add_argument("--verbose",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_file = None
    if args.verbose:
        log_file = "nba_predictor.log"
    setup_logging(log_file)

    if args.settle_bets:
        from infrastructure.reporting.settle_bets import settle_pending_bets
        res = settle_pending_bets(db_path=args.db_path, excel_path=args.excel_path)
        print(f"\n[Liquidación] Apuestas liquidadas: {res['settled']} (Ganadas: {res['won']}, Perdidas: {res['lost']}, Push: {res['push']})")
        print(f"[Liquidación] P&L Total: {res['total_pnl']:+.4f} unidades. Pendientes restantes: {res['pending_remaining']}\n")
    elif args.settle_manual:
        controller = build_app(
            team_csv=args.team_csv,
            history_csv=args.history_csv,
            cache_ttl_hours=args.ttl,
            excel_path=args.excel_path,
        )
        controller.prompt_manual_settlement()
    else:
        controller = build_app(
            team_csv=args.team_csv,
            history_csv=args.history_csv,
            cache_ttl_hours=args.ttl,
            excel_path=args.excel_path,
        )

        controller.run(
            home_query=args.home,
            away_query=args.away,
            season=args.season,
            force_fresh=args.fresh,
        )


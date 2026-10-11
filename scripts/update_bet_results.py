# -*- coding: utf-8 -*-
"""
scripts/update_bet_results.py
-----------------------------
Script ejecutable para auditar y liquidar apuestas pendientes en la bitácora Excel (.xlsx)
contra los marcadores finales de la base de datos histórica SQLite.

Uso:
    python scripts/update_bet_results.py
    python scripts/update_bet_results.py --excel-path "C:\\Users\\mblad\\Documents\\Codigoes\\nba_bet_tracking.xlsx"
    python scripts/update_bet_results.py --db-path "nba_predictor.db" --verbose
"""

import argparse
import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from infrastructure.config import config
from infrastructure.logging_config import setup_logging
from infrastructure.reporting.settle_bets import settle_pending_bets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoría y Liquidación Automatizada de Apuestas NBA (Settlement Ledger)"
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help="Ruta al archivo Excel (.xlsx) de seguimiento de apuestas",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Ruta a la base de datos SQLite con datos históricos (por defecto: config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilita registro detallado de logs en consola",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    excel_path = args.excel_path or getattr(config, "bet_tracking_excel_path", None)
    db_path = args.db_path or getattr(config, "database_path", "nba_predictor.db")

    print("\n" + "=" * 65)
    print("  NBA BET SETTLEMENT ENGINE — LIQUIDACIÓN DE APUESTAS")
    print("=" * 65)
    print(f"  Archivo Excel:   {excel_path or 'Default'}")
    print(f"  Base de Datos:   {db_path}")
    print("-" * 65)

    summary = settle_pending_bets(
        db_path=db_path,
        excel_path=excel_path,
    )

    if summary["settled"] == 0:
        msg = summary.get("message", "No se encontraron partidos finalizados para apuestas pendientes.")
        print(f"\n  ℹ️  {msg}")
        print(f"  Apuestas aún pendientes: {summary['pending_remaining']}")
        print("=" * 65 + "\n")
        return

    print(f"\n  ✅ Apuestas liquidadas con éxito: {summary['settled']}")
    print("-" * 65)
    print(f"  {'PARTIDO':<30} {'SELECCIÓN':<16} {'MARCADOR':<10} {'ESTADO':<9} {'P&L'}")
    print("-" * 65)

    for item in summary.get("details", []):
        matchup = item["matchup"][:28]
        selection = str(item["selection"])[:14]
        score = item["score"]
        status = item["status"]
        pnl = f"{item['pnl']:+.4f} u"
        print(f"  {matchup:<30} {selection:<16} {score:<10} {status:<9} {pnl}")

    print("-" * 65)
    print(f"  Ganadas:     {summary['won']:2d}")
    print(f"  Perdidas:    {summary['lost']:2d}")
    print(f"  Push:        {summary['push']:2d}")
    print(f"  P&L Neto:    {summary['total_pnl']:+.4f} unidades")
    print(f"  Pendientes:  {summary['pending_remaining']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

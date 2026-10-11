# -*- coding: utf-8 -*-
"""
scripts/build_training_dataset.py
---------------------------------
Script para transformar los datos históricos de SQLite en el dataset de entrenamiento
enriquecido con métricas canónicas por posesión, Four Factors de Dean Oliver,
EWMA temporal sin fuga de datos y variables de fatiga/descanso.

Uso:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --output-format both --halflife 12.0
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Permitir ejecución directa del script agregando el directorio raíz al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import structlog

from domain.analytics.feature_builder import FeatureBuilder
from infrastructure.database import Database

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    """Configura los argumentos por CLI."""
    parser = argparse.ArgumentParser(
        description="Generación de dataset de entrenamiento analítico de la NBA con EWMA y Four Factors."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Ruta a la base de datos SQLite (default: nba_predictor.db o config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directorio de destino para los archivos generados (default: data).",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv", "both"],
        default="both",
        help="Formato de exportación del dataset (default: both).",
    )
    parser.add_argument(
        "--halflife",
        type=float,
        default=12.0,
        help="Parámetro half-life para EWMA en número de partidos (default: 12.0).",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Partidos mínimos previos para incluir un juego en el dataset (default: 5).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n=======================================================")
    print(" NBA Feature Engineering & Training Dataset Builder")
    print("=======================================================")
    print(f" Parámetro EWMA Half-life: {args.halflife} partidos")
    print(f" Mínimo de partidos previos: {args.min_games}")
    print(f" Formato de salida: {args.output_format.upper()}")

    db = Database(args.db_path) if args.db_path else Database()
    builder = FeatureBuilder(db=db, halflife=args.halflife, min_games=args.min_games)

    print("\n-> Extrayendo datos y calculando métricas avanzadas...")
    df = builder.build_dataset_pipeline()

    if df.empty:
        print("\n[!] No se encontraron suficientes datos históricos en la base de datos.")
        print("    Asegúrate de ejecutar primero la ingesta:")
        print("    python scripts/fetch_historical_data.py --seasons 2022-23 2023-24\n")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "processed_features.parquet"
    csv_path = output_dir / "processed_features.csv"

    if args.output_format in ["parquet", "both"]:
        try:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f" [OK] Dataset guardado en Parquet: {parquet_path} ({len(df):,} filas)")
        except Exception as e:
            logger.warning("no-se-pudo-guardar-parquet", error=str(e))
            print(f" [!] Error al exportar Parquet ({e}). Usando CSV como alternativa.")
            df.to_csv(csv_path, index=False)
            print(f" [OK] Dataset guardado en CSV: {csv_path} ({len(df):,} filas)")

    if args.output_format in ["csv", "both"] and (args.output_format == "csv" or not parquet_path.exists()):
        df.to_csv(csv_path, index=False)
        print(f" [OK] Dataset guardado en CSV: {csv_path} ({len(df):,} filas)")
    elif args.output_format == "both":
        df.to_csv(csv_path, index=False)
        print(f" [OK] Dataset guardado en CSV: {csv_path} ({len(df):,} filas)")

    # Resumen y correlaciones clave con point_spread
    print("\n-------------------------------------------------------")
    print(" RESUMEN DEL DATASET GENERADO:")
    print(f" - Partidos analizados: {len(df):,}")
    print(f" - Total de características: {len(df.columns)}")
    print(f" - Victorias locales (Home Win Rate): {df['home_win'].mean():.2%}")
    print(f" - Promedio diferencial de puntos (Spread): {df['point_spread'].mean():+.2f}")

    key_diff_cols = [
        "diff_net_rtg",
        "diff_ortg",
        "diff_drtg",
        "diff_efg_pct",
        "diff_tov_pct",
        "diff_orb_pct",
        "diff_drb_pct",
        "rest_advantage",
    ]
    available_diffs = [col for col in key_diff_cols if col in df.columns]

    if available_diffs:
        print("\n Correlación con el Diferencial Real de Puntos (point_spread):")
        corrs = df[available_diffs].apply(lambda c: c.corr(df["point_spread"]))
        for col, val in corrs.sort_values(ascending=False).items():
            print(f"   * {col:<18}: {val:+.4f}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()

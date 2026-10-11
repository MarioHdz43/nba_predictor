# -*- coding: utf-8 -*-
"""
scripts/train_model.py
----------------------
Pipeline cuantitativo de entrenamiento sobre Point Spread (margen de victoria)
para baloncesto profesional (NBA).

Características principales:
1. Target primario continuo: point_spread = home_score - away_score.
2. Validación Walk-Forward (Expanding Window) sin filtración temporal.
3. Estimación con LightGBM Regressor (fallback a XGBoost).
4. Calibración probabilística mediante CDF normal empírica: P(Home Win) = Phi(mu_MOV / sigma_MOV).
5. Métricas completas: MAE, RMSE, Accuracy, Brier Score y Log-Loss.
6. Serialización en models/ junto con metadatos de calibración.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
import structlog

from domain.analytics.beta_calibration import BetaCalibrator

logger = structlog.get_logger()

# Detección de modelos de Gradient Boosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# Features seleccionadas para el modelo de margen
FEATURE_COLUMNS = [
    # Métricas diferenciales clave (Home - Away)
    "diff_net_rtg",
    "diff_ortg",
    "diff_drtg",
    "diff_efg_pct",
    "diff_tov_pct",
    "diff_orb_pct",
    "diff_drb_pct",
    "diff_ft_rate",
    "diff_opp_efg_pct",
    "diff_opp_tov_pct",
    "diff_opp_ft_rate",
    "diff_poss",
    "diff_pts",
    "diff_opp_pts",
    # Niveles base de local y visitante
    "home_ewma_net_rtg",
    "away_ewma_net_rtg",
    "home_ewma_ortg",
    "away_ewma_ortg",
    "home_ewma_drtg",
    "away_ewma_drtg",
    "home_ewma_efg_pct",
    "away_ewma_efg_pct",
    # Fatiga y descanso
    "rest_advantage",
    "home_rest_days",
    "away_rest_days",
    "home_is_b2b",
    "away_is_b2b",
    # Contexto geográfico y viajes
    "is_altitude_advantage",
    "travel_miles_away",
    "time_zones_crossed_away",
    "travel_fatigue_away",
]


def create_base_regressor(model_type: str = "lightgbm", random_state: int = 42) -> Any:
    """Instancia un regresor con regularización contra sobreajuste."""
    if model_type == "lightgbm" and HAS_LIGHTGBM:
        return lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif HAS_XGBOOST:
        return xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        )


def convert_spread_to_probability(
    predicted_spread: np.ndarray | float,
    sigma_mov: float,
) -> np.ndarray | float:
    """
    Convierte el margen proyectado a probabilidad de victoria mediante la CDF Normal:
        P(Home Win) = Phi(mu_MOV / sigma_MOV)
    """
    sigma = max(sigma_mov, 1.0)
    z_score = np.asarray(predicted_spread, dtype=float) / sigma
    probs = norm.cdf(z_score)
    # Acotar en rango realista [0.05, 0.95]
    probs = np.clip(probs, 0.05, 0.95)
    return float(probs) if np.isscalar(predicted_spread) else probs


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "point_spread",
    model_type: str = "lightgbm",
    return_predictions: bool = False,
) -> Any:
    """
    Realiza una validación Walk-Forward (Expanding Window) cronológica.
    Divide por temporadas si hay más de una, o en splits cronológicos si hay una sola.
    """
    # Orden cronológico estricto
    df = df.sort_values(by=["game_date"]).reset_index(drop=True)

    seasons = list(df["season"].dropna().unique()) if "season" in df.columns else []
    all_y_true = []
    all_y_pred = []
    fold_reports = []

    if len(seasons) >= 3:
        # Walk-forward por temporadas: Entrenar en 1..k-1, evaluar en k
        for k in range(2, len(seasons)):
            train_seasons = seasons[:k]
            test_season = seasons[k]

            train_df = df[df["season"].isin(train_seasons)]
            test_df = df[df["season"] == test_season]

            X_tr = train_df[feature_cols]
            y_tr = train_df[target_col]
            X_te = test_df[feature_cols]
            y_te = test_df[target_col]

            model = create_base_regressor(model_type)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            all_y_true.extend(y_te.values)
            all_y_pred.extend(preds)

            fold_mae = mean_absolute_error(y_te, preds)
            fold_rmse = np.sqrt(mean_squared_error(y_te, preds))
            fold_reports.append({
                "test_season": test_season,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mae": round(fold_mae, 4),
                "rmse": round(fold_rmse, 4),
            })
    else:
        # Walk-forward en splits crecientes si los datos son de 1 o 2 temporadas
        n_splits = 4
        total_rows = len(df)
        step = total_rows // (n_splits + 1)

        for i in range(1, n_splits + 1):
            train_end = step * i
            test_end = step * (i + 1) if i < n_splits else total_rows

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            if len(test_df) == 0:
                continue

            X_tr = train_df[feature_cols]
            y_tr = train_df[target_col]
            X_te = test_df[feature_cols]
            y_te = test_df[target_col]

            model = create_base_regressor(model_type)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            all_y_true.extend(y_te.values)
            all_y_pred.extend(preds)

            fold_reports.append({
                "split": i,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mae": round(mean_absolute_error(y_te, preds), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_te, preds)), 4),
            })

    if not all_y_true:
        # Fallback si no hay suficientes datos para splits
        X = df[feature_cols]
        y = df[target_col]
        model = create_base_regressor(model_type)
        model.fit(X, y)
        preds = model.predict(X)
        all_y_true = y.values
        all_y_pred = preds

    all_y_true = np.asarray(all_y_true)
    all_y_pred = np.asarray(all_y_pred)

    overall_mae = float(mean_absolute_error(all_y_true, all_y_pred))
    overall_rmse = float(np.sqrt(mean_squared_error(all_y_true, all_y_pred)))

    # Conversión a clasificación y métricas probabilísticas
    home_win_actual = (all_y_true > 0).astype(int)
    home_win_probs = convert_spread_to_probability(all_y_pred, sigma_mov=overall_rmse)
    home_win_preds = (home_win_probs > 0.5).astype(int)

    acc = float(accuracy_score(home_win_actual, home_win_preds))
    brier = float(brier_score_loss(home_win_actual, home_win_probs))
    loss = float(log_loss(home_win_actual, home_win_probs))

    metrics_summary = {
        "mae": round(overall_mae, 4),
        "rmse": round(overall_rmse, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "samples_evaluated": len(all_y_true),
    }

    if return_predictions:
        return metrics_summary, fold_reports, all_y_true, all_y_pred
    return metrics_summary, fold_reports


def train_and_save_model(
    data_path: Path,
    output_dir: Path,
    model_type: str = "lightgbm",
) -> Dict[str, Any]:
    """Carga los datos, ejecuta Walk-Forward, entrena el modelo final y lo serializa."""
    logger.info("cargando-dataset-features", path=str(data_path))

    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Filtrar solo columnas presentes en el dataset
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if len(available_features) < 5:
        raise ValueError(f"El dataset contiene muy pocas features compatibles: {available_features}")

    # Eliminar filas con nulos en las features seleccionadas o en el target
    clean_df = df.dropna(subset=available_features + ["point_spread"]).copy()
    logger.info("dataset-limpio", total_rows=len(clean_df), features_count=len(available_features))

    # 1. Validación Walk-Forward
    print("\n-> Ejecutando Validación Walk-Forward (Expanding Window)...")
    val_metrics, fold_reports, all_y_true, all_y_pred = walk_forward_validation(
        clean_df,
        feature_cols=available_features,
        target_col="point_spread",
        model_type=model_type,
        return_predictions=True,
    )

    # Ajustar calibrador Beta con las probabilidades de validación Walk-Forward
    val_probs = convert_spread_to_probability(all_y_pred, sigma_mov=val_metrics["rmse"])
    y_true_binary = (all_y_true > 0).astype(int)
    beta_calibrator = BetaCalibrator().fit(val_probs, y_true_binary)
    calibrated_val_probs = beta_calibrator.predict_proba(val_probs)
    calibrated_brier = float(brier_score_loss(y_true_binary, calibrated_val_probs))
    val_metrics["brier_score_calibrated"] = round(calibrated_brier, 4)

    print("\n-------------------------------------------------------")
    print(" RESULTADOS DE VALIDACIÓN WALK-FORWARD:")
    print(f" - MAE Diferencial:      {val_metrics['mae']:.2f} pts")
    print(f" - RMSE Diferencial:     {val_metrics['rmse']:.2f} pts (sigma_MOV)")
    print(f" - Accuracy Ganador:     {val_metrics['accuracy']:.2%}")
    print(f" - Brier Score (Bruto):  {val_metrics['brier_score']:.4f}")
    print(f" - Brier (Beta Calib):   {val_metrics['brier_score_calibrated']:.4f}")
    print(f" - Log-Loss:             {val_metrics['log_loss']:.4f}")
    print("-------------------------------------------------------")

    # 2. Entrenamiento del modelo final sobre el 100% de los datos procesados
    print("\n-> Entrenando modelo final con el histórico completo...")
    final_model = create_base_regressor(model_type)
    final_model.fit(clean_df[available_features], clean_df["point_spread"])

    # 3. Exportar artefactos a models/
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "nba_spread_model.joblib"
    metadata_path = output_dir / "model_metadata.json"

    model_artifact = {
        "model": final_model,
        "feature_names": available_features,
        "sigma_mov": val_metrics["rmse"],
        "model_type": model_type,
        "beta_calibrator": beta_calibrator,
        "validation_metrics": val_metrics,
        "trained_samples": len(clean_df),
    }

    joblib.dump(model_artifact, model_path)
    logger.info("modelo-guardado", path=str(model_path))

    metadata = {
        "model_path": str(model_path.name),
        "model_type": model_type,
        "sigma_mov": val_metrics["rmse"],
        "features_count": len(available_features),
        "features": available_features,
        "beta_calibration": beta_calibrator.to_dict(),
        "validation_metrics": val_metrics,
        "fold_reports": fold_reports,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Artefacto guardado exitosamente en: {model_path}")
    print(f"[OK] Metadatos guardados en: {metadata_path}\n")

    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Configura los argumentos por CLI."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo predictivo sobre Point Spread con Walk-Forward Validation."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Ruta al archivo processed_features (.parquet o .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio de destino para los modelos entrenados (default: models).",
    )
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "xgboost"],
        default="lightgbm" if HAS_LIGHTGBM else "xgboost",
        help="Algoritmo de Gradient Boosting a utilizar.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Buscar automáticamente el dataset de features si no se especificó
    if args.features_path:
        data_path = Path(args.features_path)
    else:
        parquet_cand = Path("data/processed_features.parquet")
        csv_cand = Path("data/processed_features.csv")
        if parquet_cand.exists():
            data_path = parquet_cand
        elif csv_cand.exists():
            data_path = csv_cand
        else:
            print("[!] No se encontró ningún dataset en data/processed_features.[parquet|csv].")
            print("    Ejecuta primero: python scripts/build_training_dataset.py")
            return

    output_dir = Path(args.output_dir)
    train_and_save_model(data_path=data_path, output_dir=output_dir, model_type=args.model_type)


if __name__ == "__main__":
    main()

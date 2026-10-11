# -*- coding: utf-8 -*-
"""
domain/engines/ml_engine.py
---------------------------
Motor de Machine Learning para predicción de partidos de la NBA.
Consume modelos entrenados sobre Point Spread (LightGBM/XGBoost Regressor),
aplica calibración probabilística con la CDF Normal empírica:
    P(Home Win) = Phi(mu_MOV / sigma_MOV)
y calcula los factores clave del partido a partir de métricas canónicas avanzadas.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
import structlog

from domain.analytics.advanced_metrics import (
    calculate_possessions,
    calculate_offensive_rating,
    calculate_defensive_rating,
    calculate_net_rating,
    calculate_efg_pct,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_drb_pct,
    calculate_ft_rate,
)
from domain.entities import H2HStats, PredictionResult, Team, TeamStats
from domain.prediction_engine import PredictionEngine
from infrastructure.config import config

logger = structlog.get_logger()


class MLEngine(PredictionEngine):
    """
    Motor de predicción cuantitativo basado en modelos de Machine Learning (Point Spread).
    """

    DEFAULT_FEATURE_NAMES = [
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
        "home_ewma_net_rtg",
        "away_ewma_net_rtg",
        "home_ewma_ortg",
        "away_ewma_ortg",
        "home_ewma_drtg",
        "away_ewma_drtg",
        "home_ewma_efg_pct",
        "away_ewma_efg_pct",
        "rest_advantage",
        "home_rest_days",
        "away_rest_days",
        "home_is_b2b",
        "away_is_b2b",
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._feature_names: List[str] = self.DEFAULT_FEATURE_NAMES
        self._sigma_mov: float = 12.5  # RMSE residual empírico típico en la NBA
        self._is_regression: bool = True
        self._is_trained: bool = False
        self._beta_calibrator: Optional[Any] = None

        self._load_artifact()

    def _find_candidate_paths(self) -> List[Path]:
        candidates = []
        if self._model_path:
            candidates.append(Path(self._model_path))

        configured = config._config.get("prediction", {}).get("model_path")
        if configured:
            candidates.append(Path(configured))

        candidates.extend([
            Path("models/nba_spread_model.joblib"),
            Path("models/nba_spread_model.json"),
            Path(__file__).parent.parent.parent / "models" / "nba_spread_model.joblib",
            Path("xgboost_model.json"),
        ])
        return candidates

    def _load_artifact(self) -> bool:
        """Intenta cargar el artefacto entrenado más reciente."""
        for path in self._find_candidate_paths():
            if path.exists():
                try:
                    if str(path).endswith(".joblib"):
                        data = joblib.load(path)
                        if isinstance(data, dict):
                            self._model = data.get("model")
                            self._feature_names = data.get("feature_names", self.DEFAULT_FEATURE_NAMES)
                            self._sigma_mov = float(data.get("sigma_mov", 12.5))
                            self._beta_calibrator = data.get("beta_calibrator")
                            self._is_regression = True
                        else:
                            self._model = data
                        self._is_trained = True
                        logger.info("ml-modelo-cargado", path=str(path), sigma_mov=self._sigma_mov, has_calibrator=self._beta_calibrator is not None)
                        return True
                    elif str(path).endswith(".json"):
                        import xgboost as xgb
                        clf = xgb.XGBClassifier()
                        clf.load_model(str(path))
                        self._model = clf
                        self._is_regression = False
                        self._is_trained = True
                        logger.info("xgboost-json-cargado", path=str(path))
                        return True
                except Exception as e:
                    logger.warning("no-se-pudo-cargar-modelo", path=str(path), error=str(e))
        return False

    def is_trained(self) -> bool:
        return self._is_trained

    def get_engine_name(self) -> str:
        return "ml_point_spread"

    def get_features_used(self) -> List[str]:
        return self._feature_names

    def _extract_vector(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_team: Optional[Team],
        away_team: Optional[Team],
        h2h_stats: Optional[H2HStats],
    ) -> Dict[str, float]:
        """Calcula el diccionario de métricas avanzadas y diferenciales para el partido."""
        h = home_stats.to_dict()
        a = away_stats.to_dict()

        # Posesiones
        poss_h = calculate_possessions(h.get("FGA", 85.0), h.get("FTA", 20.0), h.get("OREB", 9.0), h.get("TOV", 12.0))
        poss_a = calculate_possessions(a.get("FGA", 85.0), a.get("FTA", 20.0), a.get("OREB", 9.0), a.get("TOV", 12.0))
        poss_game = 0.5 * (poss_h + poss_a)

        # Ratings
        ortg_h = calculate_offensive_rating(h.get("PTS", 110.0), poss_game)
        drtg_h = calculate_defensive_rating(h.get("OPP_PTS", a.get("PTS", 108.0)), poss_game)
        net_rtg_h = calculate_net_rating(ortg_h, drtg_h)

        ortg_a = calculate_offensive_rating(a.get("PTS", 108.0), poss_game)
        drtg_a = calculate_defensive_rating(a.get("OPP_PTS", h.get("PTS", 110.0)), poss_game)
        net_rtg_a = calculate_net_rating(ortg_a, drtg_a)

        # Four factors
        efg_h = calculate_efg_pct(h.get("FGM", 40.0), h.get("FG3M", 11.0), h.get("FGA", 85.0))
        tov_h = calculate_tov_pct(h.get("TOV", 12.0), h.get("FGA", 85.0), h.get("FTA", 20.0))
        orb_h = calculate_orb_pct(h.get("OREB", 9.0), a.get("DREB", 33.0))
        drb_h = calculate_drb_pct(h.get("DREB", 33.0), a.get("OREB", 9.0))
        ft_h = calculate_ft_rate(h.get("FTM", 16.0), h.get("FGA", 85.0))

        efg_a = calculate_efg_pct(a.get("FGM", 39.0), a.get("FG3M", 10.0), a.get("FGA", 85.0))
        tov_a = calculate_tov_pct(a.get("TOV", 13.0), a.get("FGA", 85.0), a.get("FTA", 20.0))
        orb_a = calculate_orb_pct(a.get("OREB", 9.0), h.get("DREB", 33.0))
        drb_a = calculate_drb_pct(a.get("DREB", 33.0), h.get("OREB", 9.0))
        ft_a = calculate_ft_rate(a.get("FTM", 15.0), a.get("FGA", 85.0))

        # Rest y fatiga
        rest_h = 3.0
        is_b2b_h = 0.0
        rest_a = 2.0
        is_b2b_a = 0.0

        vector: Dict[str, float] = {
            "diff_net_rtg": net_rtg_h - net_rtg_a,
            "diff_ortg": ortg_h - ortg_a,
            "diff_drtg": drtg_h - drtg_a,
            "diff_efg_pct": efg_h - efg_a,
            "diff_tov_pct": tov_h - tov_a,
            "diff_orb_pct": orb_h - orb_a,
            "diff_drb_pct": drb_h - drb_a,
            "diff_ft_rate": ft_h - ft_a,
            "diff_opp_efg_pct": efg_a - efg_h,
            "diff_opp_tov_pct": tov_a - tov_h,
            "diff_opp_ft_rate": ft_a - ft_h,
            "diff_poss": poss_h - poss_a,
            "diff_pts": h.get("PTS", 110.0) - a.get("PTS", 108.0),
            "diff_opp_pts": drtg_h - drtg_a,
            "home_ewma_net_rtg": net_rtg_h,
            "away_ewma_net_rtg": net_rtg_a,
            "home_ewma_ortg": ortg_h,
            "away_ewma_ortg": ortg_a,
            "home_ewma_drtg": drtg_h,
            "away_ewma_drtg": drtg_a,
            "home_ewma_efg_pct": efg_h,
            "away_ewma_efg_pct": efg_a,
            "rest_advantage": rest_h - rest_a,
            "home_rest_days": rest_h,
            "away_rest_days": rest_a,
            "home_is_b2b": is_b2b_h,
            "away_is_b2b": is_b2b_a,
        }
        return vector

    def predict(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_name: str,
        away_name: str,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
        h2h_stats: Optional[H2HStats] = None,
    ) -> PredictionResult:
        vector = self._extract_vector(home_stats, away_stats, home_team, away_team, h2h_stats)
        feature_vals = [vector.get(col, 0.0) for col in self._feature_names]

        projected_spread: Optional[float] = None
        key_factors: List[str] = []

        projected_margin: Optional[float] = None
        if self._is_trained and self._model is not None:
            try:
                X_input = pd.DataFrame([feature_vals], columns=self._feature_names)
                if self._is_regression:
                    predicted_margin = float(self._model.predict(X_input)[0])
                    projected_margin = round(predicted_margin, 2)
                    projected_spread = predicted_margin
                    # P(Home Win) = Phi(mu / sigma)
                    raw_prob = float(norm.cdf(predicted_margin / self._sigma_mov))
                    if self._beta_calibrator is not None:
                        prob = float(self._beta_calibrator.predict_proba(raw_prob))
                        key_factors.append(
                            f"Margen proyectado: {predicted_margin:+.1f} pts (calibrado vía Beta Calibration)"
                        )
                    else:
                        prob = raw_prob
                        key_factors.append(
                            f"Margen proyectado: {predicted_margin:+.1f} pts (sigma={self._sigma_mov:.1f})"
                        )
                else:
                    raw_prob = float(self._model.predict_proba(X_input)[0][1])
                    if self._beta_calibrator is not None:
                        prob = float(self._beta_calibrator.predict_proba(raw_prob))
                    else:
                        prob = raw_prob
            except Exception as e:
                logger.warning("fallo-inferencia-ml-usando-fallback", error=str(e))
                prob = self._heuristic_fallback(vector)
        else:
            prob = self._heuristic_fallback(vector)

        # Aplicar límites de configuración
        min_p = getattr(config, "min_probability", 0.10)
        max_p = getattr(config, "max_probability", 0.90)
        home_win_prob = max(min_p, min(max_p, prob))
        away_win_prob = 1.0 - home_win_prob

        # Confianza basada en certeza probabilística
        confidence = min(0.95, max(0.50, abs(home_win_prob - 0.5) * 2.0 + 0.45))

        # Factores analíticos clave
        net_diff = vector.get("diff_net_rtg", 0.0)
        efg_diff = vector.get("diff_efg_pct", 0.0)
        key_factors.append(f"Diferencial Net Rating: {net_diff:+.1f}")
        key_factors.append(f"Diferencial eFG%: {efg_diff:+.1%}")
        if vector.get("rest_advantage", 0.0) != 0.0:
            key_factors.append(f"Ventaja descanso: {vector.get('rest_advantage'):+.0f} dias")

        return PredictionResult(
            home_team=home_name,
            away_team=away_name,
            home_win_prob=round(home_win_prob, 4),
            away_win_prob=round(away_win_prob, 4),
            confidence=round(confidence, 4),
            key_factors=key_factors,
            projected_margin=projected_margin,
        )

    def _heuristic_fallback(self, vector: Dict[str, float]) -> float:
        """Fallback analítico basado en Net Rating y Four Factors si no hay modelo entrenado."""
        net_diff = vector.get("diff_net_rtg", 0.0)
        # En la NBA, cada punto de Net Rating diferencial equivale a ~2.7% de win probability
        # más ventaja de local de 55% (~+0.05 sobre 0.5)
        prob = 0.50 + 0.05 + (net_diff * 0.027)
        return float(np.clip(prob, 0.10, 0.90))

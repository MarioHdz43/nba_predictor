# -*- coding: utf-8 -*-
"""
domain/engines/xgboost_engine.py
--------------------------------
Motor de predicción usando XGBoost (Machine Learning).
Aprende de datos históricos para hacer predicciones.

Features:
- Estadísticas de equipo (PTS, FG%, etc.)
- Forma reciente (win%)
- Lesiones
- H2H histórico
- Ventaja de	local
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from scipy.stats import norm
import xgboost as xgb

from domain.entities import (
    H2HStats,
    PredictionResult,
    Team,
    TeamStats,
)
from domain.prediction_engine import PredictionEngine
from infrastructure.config import config

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "pts_diff",
    "fg_pct_diff",
    "fg3_pct_diff",
    "reb_diff",
    "ast_diff",
    "tov_diff",
    "stl_diff",
    "blk_diff",
    "plus_minus_diff",
    "home_win_pct",
    "away_win_pct",
    "home_injury_impact",
    "away_injury_impact",
    "h2h_home_win_pct",
    "home_advantage",
]


class XGBoostEngine(PredictionEngine):
    """
    Motor de predicción basado en Machine Learning (XGBoost / LightGBM).
    Soporta modelos de regresión sobre margen (Point Spread) con calibración normal
    y modelos de clasificación directos.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model: Optional[Any] = None
        self._model_path = model_path or "xgboost_model.json"
        self._is_regression: bool = False
        self._sigma_mov: float = 12.5
        self._is_trained = self._load_model()

        if not self._is_trained:
            logger.warning("Modelo ML no encontrado. Usando modo fallback heurístico.")
            self._model = self._create_fallback_model()

    def _load_model(self) -> bool:
        # Prioridad 1: Modelo entrenado moderno de Point Spread (.joblib)
        spread_candidates = [
            Path("models/nba_spread_model.joblib"),
            Path(__file__).parent.parent.parent / "models" / "nba_spread_model.joblib",
        ]
        for sp in spread_candidates:
            if sp.exists():
                try:
                    artifact = joblib.load(sp)
                    if isinstance(artifact, dict):
                        self._model = artifact.get("model")
                        self._sigma_mov = float(artifact.get("sigma_mov", 12.5))
                    else:
                        self._model = artifact
                    self._is_regression = True
                    logger.info("Modelo Point Spread cargado desde %s", sp)
                    return True
                except Exception as e:
                    logger.warning("Error cargando modelo joblib: %s", e)

        # Prioridad 2: Modelo XGBoost JSON clásico
        if not Path(self._model_path).exists():
            return False
        try:
            self._model = xgb.XGBClassifier()
            self._model.load_model(self._model_path)
            self._is_regression = False
            logger.info("Modelo XGBoost cargado desde %s", self._model_path)
            return True
        except Exception as e:
            logger.warning("No se pudo cargar modelo: %s", e)
            return False

    def _create_fallback_model(self) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        return model

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
        features = self._extract_features(
            home_stats, away_stats, home_team, away_team, h2h_stats
        )

        key_factors: List[str] = []

        if self._is_trained and self._model:
            if self._is_regression:
                margin = float(self._model.predict([features[:getattr(self._model, "n_features_in_", len(features))]])[0])
                prob = float(norm.cdf(margin / self._sigma_mov))
                key_factors.append(f"Margen proyectado: {margin:+.1f} pts")
            elif hasattr(self._model, "predict_proba"):
                prob = float(self._model.predict_proba([features])[0][1])
            else:
                prob = self._heuristic_predict(features)
        else:
            prob = self._heuristic_predict(features)

        min_prob = config.min_probability
        max_prob = config.max_probability
        home_win_prob = max(min_prob, min(max_prob, prob))
        away_win_prob = 1.0 - home_win_prob

        confidence = self._calculate_confidence(features, home_win_prob)

        additional_factors = self._explain_prediction(
            features, home_name, away_name, home_team, away_team
        )
        key_factors.extend(additional_factors)

        return PredictionResult(
            home_team=home_name,
            away_team=away_name,
            home_win_prob=round(home_win_prob, 4),
            away_win_prob=round(away_win_prob, 4),
            confidence=round(confidence, 4),
            key_factors=key_factors,
        )

    def _extract_features(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_team: Optional[Team],
        away_team: Optional[Team],
        h2h_stats: Optional[H2HStats],
    ) -> List[float]:
        home_dict = home_stats.to_dict()
        away_dict = away_stats.to_dict()

        features = []
        for stat in ["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "STL", "BLK", "PLUS_MINUS"]:
            home_val = home_dict.get(stat, 0.0)
            away_val = away_dict.get(stat, 0.0)
            diff = home_val - away_val
            features.append(diff)

        if home_team and home_team.recent_form:
            features.append(home_team.recent_form.win_pct)
        else:
            features.append(0.5)

        if away_team and away_team.recent_form:
            features.append(away_team.recent_form.win_pct)
        else:
            features.append(0.5)

        home_injury = home_team.injury_adjustment if home_team else 0.0
        away_injury = away_team.injury_adjustment if away_team else 0.0
        features.append(home_injury)
        features.append(away_injury)

        if h2h_stats and h2h_stats.sample_size_ok:
            features.append(h2h_stats.home_win_pct)
        else:
            features.append(0.5)

        features.append(config.home_advantage)

        return features

    def _heuristic_predict(self, features: List[float]) -> float:
        weights = [
            0.25, 0.15, 0.10, 0.10, 0.08, -0.05, 0.05, 0.05, 0.12,
            0.05, 0.05, 0.02, 0.02, 0.03, 0.03
        ]
        score = 0.5
        for i, w in enumerate(weights):
            if i < len(features):
                score += features[i] * w * 0.01
        return max(0.1, min(0.9, score))

    def _calculate_confidence(self, features: List[float], prob: float) -> float:
        base = 0.6
        prob_diff = abs(prob - 0.5) * 2
        return min(0.95, base + prob_diff * 0.2)

    def _explain_prediction(
        self,
        features: List[float],
        home_name: str,
        away_name: str,
        home_team: Optional[Team],
        away_team: Optional[Team],
    ) -> List[str]:
        factors = []

        if len(features) >= 1:
            pts_diff = features[0]
            if abs(pts_diff) > 5:
                better = home_name if pts_diff > 0 else away_name
                factors.append(f"Diferencia de puntos: {better} ({pts_diff:+.1f})")

        if home_team and home_team.recent_form:
            if home_team.recent_form.is_hot:
                factors.append(f"Forma caliente: {home_name}")
            elif home_team.recent_form.is_cold:
                factors.append(f"Forma fría: {home_team}")

        if away_team and away_team.recent_form:
            if away_team.recent_form.is_hot:
                factors.append(f"Forma caliente: {away_name}")

        if home_team and home_team.has_key_injuries:
            injured = [p.name for p in home_team.injured_players if p.is_starter]
            factors.append(f"Lesionados {home_name}: {', '.join(injured)}")
        if away_team and away_team.has_key_injuries:
            injured = [p.name for p in away_team.injured_players if p.is_starter]
            factors.append(f"Lesionados {away_name}: {', '.join(injured)}")

        factors.append(f"Ventaja local: {home_name} (+{config.home_advantage*100:.0f}%)")

        if self._is_trained:
            factors.append("Predicción: XGBoost ML")
        else:
            factors.append("Predicción: XGBoost (modo fallback)")

        return factors if len(factors) > 1 else ["Sin factores significativos"]

    def get_engine_name(self) -> str:
        return "xgboost"

    def get_features_used(self) -> List[str]:
        return FEATURE_NAMES

    def train(
        self,
        X: List[List[float]],
        y: List[int],
    ) -> Dict[str, float]:
        """Entrena el modelo XGBoost con datos históricos."""
        if not self._model:
            self._model = self._create_fallback_model()

        self._model.fit(X, y)

        self._model.save_model(self._model_path)
        self._is_trained = True

        train_score = self._model.score(X, y)
        return {"train_accuracy": train_score, "samples": len(X)}

    def is_trained(self) -> bool:
        return self._is_trained
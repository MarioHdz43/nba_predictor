# -*- coding: utf-8 -*-
"""
domain/engines/hybrid_engine.py
-----------------------------
Motor híbrido que combina HeuristicEngine y XGBoostEngine.
Compara ambos y retorna predicción fusionada o la mejor.
"""

from __future__ import annotations
from typing import List, Optional

from domain.entities import H2HStats, PredictionResult, Team, TeamStats
from domain.prediction_engine import PredictionEngine
from domain.engines.heuristic_engine import HeuristicEngine
from domain.engines.xgboost_engine import XGBoostEngine
from infrastructure.config import config


class HybridEngine(PredictionEngine):
    """
    Motor híbrido que combina ambos sistemas.
    
    Estrategia:
    - Si XGBoost entrenado → usa promedio ponderado
    - Si no → usa solo Heuristic
    """

    def __init__(self) -> None:
        self._heuristic = HeuristicEngine()
        self._xgboost = XGBoostEngine()

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
        result_heuristic = self._heuristic.predict(
            home_stats, away_stats, home_name, away_name,
            home_team, away_team, h2h_stats
        )

        result_ml = self._xgboost.predict(
            home_stats, away_stats, home_name, away_name,
            home_team, away_team, h2h_stats
        )

        fusion = self._fusion_predictions(result_heuristic, result_ml)

        return fusion

    def _fusion_predictions(
        self,
        h: PredictionResult,
        ml: PredictionResult,
    ) -> PredictionResult:
        if not self._xgboost.is_trained():
            return h

        ml_weight = getattr(config, "ml_weight", 0.5)

        home_prob = (
            h.home_win_prob * (1 - ml_weight) +
            ml.home_win_prob * ml_weight
        )
        away_prob = 1.0 - home_prob

        h_conf = h.confidence
        ml_conf = ml.confidence
        confidence = min(0.95, (h_conf + ml_conf) / 2 + 0.05)

        key_factors = h.key_factors.copy()
        if self._xgboost.is_trained():
            pct = int(ml_weight * 100)
            key_factors.append(f"Combinado: Heuristic + XGBoost ({pct}%)")
        else:
            key_factors.append("XGBoost (fallback)")

        return PredictionResult(
            home_team=h.home_team,
            away_team=h.away_team,
            home_win_prob=round(home_prob, 4),
            away_win_prob=round(away_prob, 4),
            confidence=round(confidence, 4),
            key_factors=key_factors,
        )

    def get_engine_name(self) -> str:
        return "hybrid"

    def get_features_used(self) -> List[str]:
        heuristic_features = self._heuristic.get_features_used()
        return heuristic_features + ["xgboost_features"]

    def get_heuristic_result(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_name: str,
        away_name: str,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
        h2h_stats: Optional[H2HStats] = None,
    ) -> PredictionResult:
        return self._heuristic.predict(
            home_stats, away_stats, home_name, away_name,
            home_team, away_team, h2h_stats
        )

    def get_xgboost_result(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_name: str,
        away_name: str,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
        h2h_stats: Optional[H2HStats] = None,
    ) -> PredictionResult:
        return self._xgboost.predict(
            home_stats, away_stats, home_name, away_name,
            home_team, away_team, h2h_stats
        )

    def is_xgboost_trained(self) -> bool:
        return self._xgboost.is_trained()
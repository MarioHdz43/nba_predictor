# -*- coding: utf-8 -*-
"""
domain/engines/heuristic_engine.py
--------------------------------
Motor de predicción heurístico basado en diferencias estadísticas ponderadas.
Usa configuración centralizada de config.yaml.
Mejora FASE 1: Recent form + Injury adjustment.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from domain.entities import H2HStats, PredictionResult, Team, TeamStats
from domain.prediction_engine import PredictionEngine
from infrastructure.config import config


class HeuristicEngine(PredictionEngine):
    """
    Motor de predicción basado en diferencias estadísticas ponderadas.
    Lee pesos y normalización desde config.yaml.

    Mejoras FASE 1:
    - Recent form weighting (momentum)
    - Injury adjustment
    """

    _LABELS: Dict[str, str] = {
        "PTS": "Puntos",
        "FG_PCT": "% Tiros de campo",
        "FG3_PCT": "% Triples",
        "REB": "Rebotes",
        "AST": "Asistencias",
        "TOV": "Pérdidas de balón",
        "STL": "Robos",
        "BLK": "Tapones",
        "PLUS_MINUS": "Plus/Minus",
    }

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
        weights = config.weights
        normalization = config.normalization
        home_advantage = config.home_advantage
        min_prob = config.min_probability
        max_prob = config.max_probability
        recent_form_weight = config.recent_form_weight
        injury_impact = config.injury_impact

        home_dict = home_stats.to_dict()
        away_dict = away_stats.to_dict()

        diffs: Dict[str, float] = {
            stat: home_dict.get(stat, 0.0) - away_dict.get(stat, 0.0)
            for stat in weights
        }

        base_score = 0.5
        for stat, weight in weights.items():
            norm = normalization.get(stat, 1.0)
            base_score += (diffs[stat] / norm) * weight

        base_score += home_advantage

        form_adjustment = self._calculate_form_adjustment(
            home_team, away_team, recent_form_weight
        )
        injury_adjustment = self._calculate_injury_adjustment(
            home_team, away_team, injury_impact
        )
        h2h_adjustment = self._calculate_h2h_adjustment(h2h_stats)

        score = base_score + form_adjustment + injury_adjustment + h2h_adjustment
        home_win_prob = max(min_prob, min(max_prob, score))
        away_win_prob = 1.0 - home_win_prob

        confidence = self._calculate_confidence(diffs, form_adjustment)
        key_factors = self._get_key_factors(
            diffs, home_name, away_name, home_team, away_team
        )

        return PredictionResult(
            home_team=home_name,
            away_team=away_name,
            home_win_prob=round(home_win_prob, 4),
            away_win_prob=round(away_win_prob, 4),
            confidence=round(confidence, 4),
            key_factors=key_factors,
        )

    def _calculate_form_adjustment(
        self,
        home_team: Optional[Team],
        away_team: Optional[Team],
        weight: float,
    ) -> float:
        if not home_team or not away_team:
            return 0.0
        if not home_team.recent_form or not away_team.recent_form:
            return 0.0

        home_form = home_team.recent_form
        away_form = away_team.recent_form

        home_win_pct = home_form.win_pct
        away_win_pct = away_form.win_pct

        diff = (home_win_pct - away_win_pct) * weight
        return diff

    def _calculate_injury_adjustment(
        self,
        home_team: Optional[Team],
        away_team: Optional[Team],
        impact_per_player: float,
    ) -> float:
        if not home_team:
            home_injury = 0.0
        else:
            home_injury = home_team.injury_adjustment

        if not away_team:
            away_injury = 0.0
        else:
            away_injury = away_team.injury_adjustment

        return (away_injury - home_injury) / 10.0

    def _calculate_h2h_adjustment(
        self,
        h2h_stats: Optional[H2HStats],
    ) -> float:
        if not h2h_stats:
            return 0.0
        if not h2h_stats.sample_size_ok:
            return 0.0
        weight = config.h2h_weight
        return h2h_stats.advantage * weight

    def _calculate_confidence(
        self,
        diffs: Dict[str, float],
        form_adjustment: float = 0.0,
    ) -> float:
        total = sum(abs(d) for d in diffs.values())
        base_conf = max(0.5, min(1.0, 0.5 + total / 50.0))
        form_bonus = abs(form_adjustment) * 0.1
        return min(1.0, base_conf + form_bonus)

    def _get_key_factors(
        self,
        diffs: Dict[str, float],
        home_name: str,
        away_name: str,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
    ) -> List[str]:
        threshold_pct = config.significant_factor_threshold
        normalization = config.normalization

        factors = []
        for stat, diff in diffs.items():
            threshold = normalization.get(stat, 1.0) * threshold_pct
            if abs(diff) > threshold:
                better_team = home_name if diff > 0 else away_name
                worse_team = away_name if diff > 0 else home_name
                label = self._LABELS.get(stat, stat)
                factors.append(
                    f"{label}: {better_team} supera a {worse_team} ({diff:+.2f})"
                )

        if home_team and home_team.recent_form:
            if home_team.recent_form.is_hot:
                factors.append(f"Forma caliente: {home_name} (últimos {home_team.recent_form.last_games} partidos)")
            elif home_team.recent_form.is_cold:
                factors.append(f"Forma fría: {home_name} (racha negativa)")
            if away_team and away_team.recent_form.is_hot:
                factors.append(f"Forma caliente: {away_name} (últimos {away_team.recent_form.last_games} partidos)")

        if home_team and home_team.has_key_injuries:
            injured = [p.name for p in home_team.injured_players if p.is_starter]
            factors.append(f"Lesionados {home_name}: {', '.join(injured)}")
        if away_team and away_team.has_key_injuries:
            injured = [p.name for p in away_team.injured_players if p.is_starter]
            factors.append(f"Lesionados {away_name}: {', '.join(injured)}")

        factors.append(f"Ventaja local: {home_name} (+{config.home_advantage*100:.0f}%)")
        return factors if len(factors) > 1 else ["Sin diferencias统计学 significativas"]

    def get_engine_name(self) -> str:
        return "heuristic"

    def get_features_used(self) -> List[str]:
        features = list(config.weights.keys())
        features.extend(["recent_form", "injury_adjustment", "h2h_history"])
        return features
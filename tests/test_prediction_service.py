# -*- coding: utf-8 -*-
"""
Tests for prediction_service.py.
Strict TDD Mode - Escribí los tests primero, después el código.
"""

import pytest

from domain import prediction_service
from domain.entities import PredictionResult, TeamStats
from domain.engines.heuristic_engine import HeuristicEngine
from domain.prediction_engine import PredictionEngine


class TestPredictionService:
    """Tests para lafachada de predicción."""

    def test_predict_returns_prediction_result(self):
        """predict() debe retornar un PredictionResult."""
        home = TeamStats(pts=110, fg_pct=0.48)
        away = TeamStats(pts=105, fg_pct=0.45)
        
        result = prediction_service.predict(
            home_stats=home,
            away_stats=away,
            home_name="Lakers",
            away_name="Celtics"
        )
        
        assert isinstance(result, PredictionResult)
        assert result.home_team == "Lakers"
        assert result.away_team == "Celtics"

    def test_predict_probabilities_sum_to_one(self):
        """Las probabilidades deben sumar 1.0."""
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        
        result = prediction_service.predict(home, away, "Home", "Away")
        
        total = result.home_win_prob + result.away_win_prob
        assert abs(total - 1.0) < 0.001

    def test_predict_probabilities_in_valid_range(self):
        """Las probabilidades deben estar entre 0 y 1."""
        home = TeamStats(pts=100)
        away = TeamStats(pts=100)
        
        result = prediction_service.predict(home, away, "A", "B")
        
        assert 0.0 <= result.home_win_prob <= 1.0
        assert 0.0 <= result.away_win_prob <= 1.0

    def test_predict_confidence_in_valid_range(self):
        """La confianza debe estar entre 0.5 y 1.0."""
        home = TeamStats(pts=110, fg_pct=0.48)
        away = TeamStats(pts=105, fg_pct=0.45)
        
        result = prediction_service.predict(home, away, "Home", "Away")
        
        assert 0.5 <= result.confidence <= 1.0

    def test_predict_with_dominant_home_team(self):
        """Equipo local superior debe tener mayor probabilidad."""
        home = TeamStats(pts=120, fg_pct=0.50, plus_minus=10)
        away = TeamStats(pts=90, fg_pct=0.40, plus_minus=-10)
        
        result = prediction_service.predict(home, away, "Lakers", "Celtics")
        
        assert result.home_win_prob > 0.5

    def test_predict_with_dominant_away_team(self):
        """Equipo visitante superior debe tener menor probabilidad local."""
        home = TeamStats(pts=90, fg_pct=0.40, plus_minus=-10)
        away = TeamStats(pts=120, fg_pct=0.50, plus_minus=10)
        
        result = prediction_service.predict(home, away, "Lakers", "Celtics")
        
        assert result.away_win_prob > 0.5

    def test_normalization_factor_returns_float(self):
        """normalization_factor() debe retornar float."""
        factor = prediction_service.normalization_factor("PTS")
        
        assert isinstance(factor, float)
        assert factor > 0

    def test_normalization_factor_unknown_stat(self):
        """unknown stat debe retornar 1.0."""
        factor = prediction_service.normalization_factor("UNKNOWN")
        
        assert factor == 1.0


class TestEngineInjection:
    """Tests para inyección de motor."""

    def test_set_engine_changes_engine(self):
        """set_engine() debe cambiar el motor."""
        new_engine = HeuristicEngine()
        
        prediction_service.set_engine(new_engine)
        
        assert prediction_service._get_engine() is new_engine

    def test_set_engine_restores_default(self):
        """Pasar None debe restaurar default."""
        prediction_service.set_engine(None)
        
        engine = prediction_service._get_engine()
        assert isinstance(engine, HeuristicEngine)
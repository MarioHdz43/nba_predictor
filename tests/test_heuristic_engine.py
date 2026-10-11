# -*- coding: utf-8 -*-
"""
Tests for HeuristicEngine.
Strict TDD Mode.
"""

import pytest

from domain.entities import H2HStats, PredictionResult, Team, TeamForm, TeamStats
from domain.engines.heuristic_engine import HeuristicEngine


class TestHeuristicEngine:
    """Tests para el motor heurístico."""

    @pytest.fixture
    def engine(self):
        return HeuristicEngine()

    def test_predict_returns_prediction_result(self, engine):
        """predict() debe retornar PredictionResult."""
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        
        result = engine.predict(home, away, "Lakers", "Celtics")
        
        assert isinstance(result, PredictionResult)
        assert result.home_team == "Lakers"

    def test_probabilities_sum_to_one(self, engine):
        """Probabilidades deben sumar 1."""
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        
        result = engine.predict(home, away, "A", "B")
        
        assert abs(result.home_win_prob + result.away_win_prob - 1.0) < 0.001

    def test_probabilities_in_range(self, engine):
        """Probabilidades entre 0 y 1."""
        home = TeamStats(pts=100)
        away = TeamStats(pts=100)
        
        result = engine.predict(home, away, "A", "B")
        
        assert 0.0 < result.home_win_prob < 1.0

    def test_confidence_in_range(self, engine):
        """Confianza entre 0.5 y 1.0."""
        home = TeamStats(pts=110, fg_pct=0.48)
        away = TeamStats(pts=105, fg_pct=0.45)
        
        result = engine.predict(home, away, "Home", "Away")
        
        assert 0.5 <= result.confidence <= 1.0

    def test_dominant_home_wins(self, engine):
        """Home mejor debe ganar."""
        home = TeamStats(pts=120, fg_pct=0.50, plus_minus=10)
        away = TeamStats(pts=90, fg_pct=0.40, plus_minus=-10)
        
        result = engine.predict(home, away, "Lakers", "Celtics")
        
        assert result.home_win_prob > 0.5

    def test_dominant_away_wins(self, engine):
        """Away mejor debe ganar."""
        home = TeamStats(pts=90, fg_pct=0.40, plus_minus=-10)
        away = TeamStats(pts=120, fg_pct=0.50, plus_minus=10)
        
        result = engine.predict(home, away, "Lakers", "Celtics")
        
        assert result.away_win_prob > 0.5

    def test_with_recent_form_hot(self, engine):
        """Forma reciente caliente debe afectar."""
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        
        home_team = Team(
            team_id=1, full_name="Lakers", abbreviation="LAL", season="2024-25",
            stats=home, recent_form=TeamForm(wins=5, losses=0, last_games=5)
        )
        
        result = engine.predict(home, away, "Lakers", "Celtics", home_team=home_team)
        
        assert result.home_win_prob > 0.5

    def test_with_injuries(self, engine):
        """Lesiones deben afectar."""
        from domain.entities import InjuredPlayer
        
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        
        home_team = Team(
            team_id=1, full_name="Lakers", abbreviation="LAL", season="2024-25",
            stats=home, injured_players=[InjuredPlayer(name="LeBron", impact=8.0, position="SF")]
        )
        
        result = engine.predict(home, away, "Lakers", "Celtics", home_team=home_team)
        
        assert isinstance(result, PredictionResult)

    def test_with_h2h_stats(self, engine):
        """H2H debe afectar."""
        home = TeamStats(pts=110)
        away = TeamStats(pts=105)
        h2h = H2HStats(home_team="Lakers", away_team="Celtics", home_wins=7, away_wins=3, total_games=10)
        
        result = engine.predict(home, away, "Lakers", "Celtics", h2h_stats=h2h)
        
        assert result.home_win_prob > 0.5

    def test_get_engine_name(self, engine):
        """get_engine_name() debe retornar string."""
        assert engine.get_engine_name() == "heuristic"

    def test_get_features_used(self, engine):
        """get_features_used() debe retornar lista."""
        features = engine.get_features_used()
        
        assert isinstance(features, list)
        assert len(features) > 0
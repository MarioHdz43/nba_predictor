# -*- coding: utf-8 -*-
"""
Tests for domain entities.
"""
from datetime import datetime, timedelta

import pytest

from domain.entities import Odds, PlayerStat, PredictionResult, Team, TeamStats


class TestTeamStats:
    def test_default_values(self):
        stats = TeamStats()
        assert stats.pts == 0.0
        assert stats.fg_pct == 0.0
        assert stats.fg3_pct == 0.0

    def test_to_dict(self):
        stats = TeamStats(pts=110.5, fg_pct=0.475, fg3_pct=0.365)
        d = stats.to_dict()
        assert d["PTS"] == 110.5
        assert d["FG_PCT"] == 0.475

    def test_from_dict(self):
        d = {"PTS": 105.0, "FG_PCT": 0.44, "FG3_PCT": 0.35, "REB": 45.0, "AST": 25.0, "TOV": 14.0, "STL": 8.0, "BLK": 5.0, "PLUS_MINUS": 3.5}
        stats = TeamStats.from_dict(d)
        assert stats.pts == 105.0
        assert stats.fg_pct == 0.44

    def test_immutability(self):
        stats = TeamStats(pts=100.0)
        with pytest.raises(AttributeError):
            stats.pts = 200.0


class TestTeam:
    def test_is_stale_without_fetched_at(self):
        team = Team(team_id=1, full_name="Test", abbreviation="TST", season="2024-25")
        assert team.is_stale() is True

    def test_is_stale_with_fresh_data(self):
        team = Team(
            team_id=1, full_name="Test", abbreviation="TST", season="2024-25", fetched_at=datetime.utcnow()
        )
        assert team.is_stale(ttl_hours=2) is False

    def test_is_stale_with_old_data(self):
        old_time = datetime.utcnow() - timedelta(hours=3)
        team = Team(team_id=1, full_name="Test", abbreviation="TST", season="2024-25", fetched_at=old_time)
        assert team.is_stale(ttl_hours=2) is True


class TestOdds:
    def test_implied_probability(self):
        odds = Odds(home_decimal=2.0, away_decimal=1.8)
        assert odds.home_implied_prob() == 0.5
        assert odds.away_implied_prob() == pytest.approx(0.5556, rel=0.01)

    def test_has_overround(self):
        odds = Odds(home_decimal=1.9, away_decimal=1.9)
        assert odds.has_overround() is True

    def test_no_overround(self):
        odds = Odds(home_decimal=2.0, away_decimal=2.0)
        assert odds.has_overround() is False


class TestPredictionResult:
    def test_attach_odds_home_win(self):
        pred = PredictionResult(
            home_team="Lakers", away_team="Celtics", home_win_prob=0.65, away_win_prob=0.35, confidence=0.75, key_factors=["Test"]
        )
        # EV = P * (odds - 1) - (1 - P) * 1
        # EV_home = 0.65 * (2.0 - 1) - 0.35 * 1 = 0.30
        # EV_away = 0.35 * (1.8 - 1) - 0.65 * 1 = -0.37
        odds = Odds(home_decimal=2.0, away_decimal=1.8)
        result = pred.attach_odds(odds)
        assert result.odds == odds
        assert result.home_expected_value == pytest.approx(0.30, rel=0.01)
        assert result.away_expected_value == pytest.approx(-0.37, rel=0.01)
        assert result.recommended_bet == "Home (Lakers)"

    def test_attach_odds_away_win(self):
        pred = PredictionResult(
            home_team="Lakers", away_team="Celtics", home_win_prob=0.35, away_win_prob=0.65, confidence=0.75, key_factors=["Test"]
        )
        # EV = P * (odds - 1) - (1 - P) * 1
        # EV_home = 0.35 * (1.8 - 1) - 0.65 * 1 = -0.37
        # EV_away = 0.65 * (2.0 - 1) - 0.35 * 1 = 0.30
        odds = Odds(home_decimal=1.8, away_decimal=2.0)
        result = pred.attach_odds(odds)
        assert result.odds == odds
        assert result.home_expected_value == pytest.approx(-0.37, rel=0.01)
        assert result.away_expected_value == pytest.approx(0.30, rel=0.01)
        assert result.recommended_bet == "Away (Celtics)"

    def test_attach_odds_no_bet_positive_ev(self):
        pred = PredictionResult(
            home_team="Lakers", away_team="Celtics", home_win_prob=0.50, away_win_prob=0.50, confidence=0.60, key_factors=["Test"]
        )
        odds = Odds(home_decimal=1.9, away_decimal=1.9)
        result = pred.attach_odds(odds)
        assert result.recommended_bet == "No bet"
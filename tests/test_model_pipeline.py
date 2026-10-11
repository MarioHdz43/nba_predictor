# -*- coding: utf-8 -*-
"""
tests/test_model_pipeline.py
----------------------------
Tests unitarios para la validación cuantitativa:
- Criterio de Kelly Fraccional (0.25 / Quarter Kelly) y Expected Value.
- Calibración de probabilidad a partir del margen proyectado (CDF Normal).
- Validación Walk-Forward (Expanding Window).
- Pipeline integral de entrenamiento, serialización e inferencia con MLEngine.
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from domain.entities import Odds, PredictionResult
from domain.engines.ml_engine import MLEngine
from scripts.train_model import (
    convert_spread_to_probability,
    create_base_regressor,
    train_and_save_model,
    walk_forward_validation,
)


class TestFinancialManagementAndKelly:
    def test_fractional_kelly_and_ev(self):
        """
        Valida el cálculo de Quarter Kelly:
            f* = max(0, (P * b - (1 - P)) / b) * 0.25, donde b = odds - 1.
        """
        pred = PredictionResult(
            home_team="Lakers",
            away_team="Celtics",
            home_win_prob=0.60,
            away_win_prob=0.40,
            confidence=0.70,
            key_factors=["Test"],
        )
        # Cuotas: Home = 2.0 (b=1.0), Away = 1.8 (b=0.8)
        # EV_home = 0.60 * 1.0 - 0.40 * 1.0 = +0.20
        # Kelly_home = (0.60 * 1.0 - 0.40) / 1.0 * 0.25 = 0.20 * 0.25 = 0.05 (5% bankroll)
        # EV_away = 0.40 * 0.8 - 0.60 * 1.0 = 0.32 - 0.60 = -0.28 (EV negativo -> Kelly = 0.0)
        odds = Odds(home_decimal=2.0, away_decimal=1.8)
        result = pred.attach_odds(odds)

        assert result.home_expected_value == pytest.approx(0.20, abs=1e-3)
        assert result.away_expected_value == pytest.approx(-0.28, abs=1e-3)
        assert result.kelly_stake_home == pytest.approx(0.05, abs=1e-3)
        assert result.kelly_stake_away == 0.0
        assert result.recommended_bet == "Home (Lakers)"

    def test_kelly_zero_when_both_negative_ev(self):
        """Si ambas cuotas ofrecen EV negativo, ambos stakes de Kelly deben ser 0.0."""
        pred = PredictionResult(
            home_team="Lakers",
            away_team="Celtics",
            home_win_prob=0.50,
            away_win_prob=0.50,
            confidence=0.60,
            key_factors=["Test"],
        )
        odds = Odds(home_decimal=1.9, away_decimal=1.9)
        result = pred.attach_odds(odds)

        assert result.home_expected_value < 0.0
        assert result.away_expected_value < 0.0
        assert result.kelly_stake_home == 0.0
        assert result.kelly_stake_away == 0.0
        assert result.recommended_bet == "No bet"


class TestNormalCalibrationCDF:
    def test_spread_to_probability_conversion(self):
        """Valida que P(Home Win) = Phi(mu / sigma)."""
        sigma = 12.0

        # Margen 0 -> 50% de probabilidad
        p_even = convert_spread_to_probability(0.0, sigma_mov=sigma)
        assert p_even == pytest.approx(0.50, abs=1e-3)

        # Margen +6 pts con sigma=12 -> z=0.5 -> Phi(0.5) ≈ 0.6915
        p_plus_6 = convert_spread_to_probability(6.0, sigma_mov=sigma)
        assert p_plus_6 == pytest.approx(0.6915, abs=1e-3)

        # Margen -6 pts con sigma=12 -> z=-0.5 -> Phi(-0.5) ≈ 0.3085
        p_minus_6 = convert_spread_to_probability(-6.0, sigma_mov=sigma)
        assert p_minus_6 == pytest.approx(0.3085, abs=1e-3)

        # Vectorizado con límites de clipping [0.05, 0.95]
        spreads = np.array([30.0, -30.0])
        probs = convert_spread_to_probability(spreads, sigma_mov=sigma)
        assert probs[0] == pytest.approx(0.95, abs=1e-3)
        assert probs[1] == pytest.approx(0.05, abs=1e-3)


class TestWalkForwardAndModelTraining:
    @pytest.fixture
    def synthetic_features_df(self):
        """Genera un dataset sintético controlado para verificar el pipeline Walk-Forward."""
        np.random.seed(42)
        n_samples = 120
        dates = pd.date_range("2022-10-01", periods=n_samples, freq="D")
        seasons = ["2022-23"] * 60 + ["2023-24"] * 60

        diff_net = np.random.normal(0, 5, n_samples)
        diff_ortg = np.random.normal(0, 4, n_samples)
        diff_drtg = np.random.normal(0, 4, n_samples)
        diff_efg = np.random.normal(0, 0.03, n_samples)
        diff_tov = np.random.normal(0, 0.02, n_samples)
        diff_orb = np.random.normal(0, 0.04, n_samples)
        diff_drb = np.random.normal(0, 0.04, n_samples)
        diff_ft = np.random.normal(0, 0.03, n_samples)
        diff_opp_efg = -diff_efg
        diff_opp_tov = -diff_tov
        diff_opp_ft = -diff_ft
        diff_poss = np.random.normal(0, 2, n_samples)
        diff_pts = diff_net * 0.8
        diff_opp_pts = -diff_pts
        rest_adv = np.random.choice([-1, 0, 1], n_samples)

        # Point spread = función de diff_net + ventaja local + ruido
        point_spread = 2.5 + (diff_net * 0.7) + (rest_adv * 1.0) + np.random.normal(0, 8, n_samples)

        data = {
            "game_id": [f"G{i:04d}" for i in range(n_samples)],
            "season": seasons,
            "game_date": [d.strftime("%Y-%m-%d") for d in dates],
            "diff_net_rtg": diff_net,
            "diff_ortg": diff_ortg,
            "diff_drtg": diff_drtg,
            "diff_efg_pct": diff_efg,
            "diff_tov_pct": diff_tov,
            "diff_orb_pct": diff_orb,
            "diff_drb_pct": diff_drb,
            "diff_ft_rate": diff_ft,
            "diff_opp_efg_pct": diff_opp_efg,
            "diff_opp_tov_pct": diff_opp_tov,
            "diff_opp_ft_rate": diff_opp_ft,
            "diff_poss": diff_poss,
            "diff_pts": diff_pts,
            "diff_opp_pts": diff_opp_pts,
            "home_ewma_net_rtg": diff_net / 2,
            "away_ewma_net_rtg": -diff_net / 2,
            "home_ewma_ortg": 110.0 + diff_ortg / 2,
            "away_ewma_ortg": 110.0 - diff_ortg / 2,
            "home_ewma_drtg": 108.0 + diff_drtg / 2,
            "away_ewma_drtg": 108.0 - diff_drtg / 2,
            "home_ewma_efg_pct": 0.53 + diff_efg / 2,
            "away_ewma_efg_pct": 0.53 - diff_efg / 2,
            "rest_advantage": rest_adv,
            "home_rest_days": np.random.choice([1, 2, 3], n_samples),
            "away_rest_days": np.random.choice([1, 2, 3], n_samples),
            "home_is_b2b": np.random.choice([0, 1], n_samples),
            "away_is_b2b": np.random.choice([0, 1], n_samples),
            "point_spread": point_spread,
            "home_win": (point_spread > 0).astype(int),
        }
        return pd.DataFrame(data)

    def test_walk_forward_validation_structure(self, synthetic_features_df):
        """Verifica que la validación Walk-Forward reporte métricas coherentes sin fuga temporal."""
        feature_cols = [
            "diff_net_rtg", "diff_ortg", "diff_drtg", "diff_efg_pct",
            "rest_advantage", "home_is_b2b", "away_is_b2b",
        ]
        metrics, fold_reports = walk_forward_validation(
            synthetic_features_df, feature_cols=feature_cols, target_col="point_spread"
        )

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "accuracy" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics

        assert metrics["mae"] > 0.0
        assert metrics["rmse"] > 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["brier_score"] >= 0.0
        assert len(fold_reports) > 0

    def test_train_and_save_model_end_to_end(self, tmp_path, synthetic_features_df):
        """Valida el pipeline completo de entrenamiento, serialización e inferencia con MLEngine."""
        data_file = tmp_path / "processed_features.parquet"
        synthetic_features_df.to_parquet(data_file, index=False)

        models_dir = tmp_path / "models"
        metadata = train_and_save_model(data_path=data_file, output_dir=models_dir)

        model_file = models_dir / "nba_spread_model.joblib"
        metadata_file = models_dir / "model_metadata.json"

        assert model_file.exists()
        assert metadata_file.exists()
        assert metadata["sigma_mov"] > 0.0

        # Verificar que MLEngine pueda cargar el artefacto y generar predicciones
        engine = MLEngine(model_path=str(model_file))
        assert engine.is_trained() is True
        assert engine._is_regression is True
        assert engine._sigma_mov == pytest.approx(metadata["sigma_mov"], abs=1e-3)

        from domain.entities import TeamStats
        h_stats = TeamStats(pts=115.0, fg_pct=0.48, fg3_pct=0.38, reb=44.0, ast=26.0, tov=12.0)
        a_stats = TeamStats(pts=105.0, fg_pct=0.44, fg3_pct=0.34, reb=40.0, ast=22.0, tov=15.0)

        pred_result = engine.predict(
            home_stats=h_stats,
            away_stats=a_stats,
            home_name="Lakers",
            away_name="Celtics",
        )

        assert isinstance(pred_result, PredictionResult)
        assert 0.0 < pred_result.home_win_prob < 1.0
        assert pred_result.home_win_prob + pred_result.away_win_prob == pytest.approx(1.0, abs=1e-3)
        assert any("Margen proyectado" in factor for factor in pred_result.key_factors)

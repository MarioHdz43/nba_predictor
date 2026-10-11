# -*- coding: utf-8 -*-
"""
tests/test_feature_engineering.py
---------------------------------
Suite de pruebas unitarias para el módulo de analítica avanzada y feature engineering.
Incluye validación matemática de fórmulas canónicas, Four Factors de Dean Oliver,
demostración estricta de Zero Data Leakage en EWMA y variables de fatiga/descanso.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from domain.analytics.advanced_metrics import (
    calculate_possessions,
    calculate_game_possessions,
    calculate_offensive_rating,
    calculate_defensive_rating,
    calculate_net_rating,
    calculate_pace,
    calculate_efg_pct,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_drb_pct,
    calculate_ft_rate,
)
from domain.analytics.feature_builder import FeatureBuilder, CANONICAL_METRICS


class TestAdvancedMetrics:
    def test_possessions_calculation(self):
        """Valida Poss = 0.5 * (FGA + 0.44 * FTA - OREB + TOV)."""
        # Escalar
        # 0.5 * (88 + 0.44 * 24 - 9 + 13) = 0.5 * (88 + 10.56 - 9 + 13) = 0.5 * 102.56 = 51.28
        poss = calculate_possessions(fga=88, fta=24, oreb=9, tov=13)
        assert poss == pytest.approx(51.28, rel=1e-3)

        # Vectorizado con Pandas
        df = pd.DataFrame({
            "fga": [88, 90],
            "fta": [24, 20],
            "oreb": [9, 10],
            "tov": [13, 12],
        })
        res = calculate_possessions(df["fga"], df["fta"], df["oreb"], df["tov"])
        assert len(res) == 2
        assert res.iloc[0] == pytest.approx(51.28, rel=1e-3)

    def test_game_possessions(self):
        """Valida que calculate_game_possessions promedie ambas estimaciones."""
        poss_game = calculate_game_possessions(
            home_fga=85, home_fta=20, home_oreb=10, home_tov=12,
            away_fga=90, away_fta=25, away_oreb=8, away_tov=14,
        )
        home_raw = 85 + 0.44 * 20 - 10 + 12  # 95.8
        away_raw = 90 + 0.44 * 25 - 8 + 14   # 107.0
        expected = 0.5 * (home_raw + away_raw)  # 101.4
        assert poss_game == pytest.approx(expected, rel=1e-3)

    def test_ratings_and_net_rating(self):
        """Valida ORtg, DRtg y NRtg con protección contra división por cero."""
        ortg = calculate_offensive_rating(pts=115, poss=100.0)
        drtg = calculate_defensive_rating(opp_pts=107, poss=100.0)
        nrtg = calculate_net_rating(ortg, drtg)

        assert ortg == 115.0
        assert drtg == 107.0
        assert nrtg == 8.0

        # División por cero protegida
        assert calculate_offensive_rating(pts=100, poss=0.0) == 0.0
        assert calculate_defensive_rating(opp_pts=100, poss=0.0) == 0.0

    def test_pace_calculation(self):
        """Valida cálculo de ritmo normalizado a 48 minutos."""
        # En tiempo regular (240 min de equipo), pace = (poss_h + poss_a) / 2
        pace_reg = calculate_pace(poss_home=100.0, poss_away=100.0, team_minutes=240.0)
        assert pace_reg == pytest.approx(100.0, rel=1e-3)

        # En tiempo extra (1 prórroga = 265 min de equipo, 53 min por jugador)
        # 48 * (100 + 100) / (2 * 53) = 9600 / 106 ≈ 90.566
        pace_ot = calculate_pace(poss_home=100.0, poss_away=100.0, team_minutes=265.0)
        assert pace_ot == pytest.approx(90.566, rel=1e-3)

    def test_four_factors_calculation(self):
        """Valida fórmulas de Dean Oliver: eFG%, TOV%, ORB%, DRB%, FT Rate."""
        # 1. eFG% = (FGM + 0.5 * 3PM) / FGA
        efg = calculate_efg_pct(fgm=40, fg3m=10, fga=80)
        assert efg == pytest.approx(0.5625, rel=1e-3)

        # 2. TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        tov_pct = calculate_tov_pct(tov=15, fga=80, fta=20)
        expected_tov = 15 / (80 + 0.44 * 20 + 15)  # 15 / 103.8 ≈ 0.144508
        assert tov_pct == pytest.approx(expected_tov, rel=1e-3)

        # 3. ORB% = OREB / (OREB + Opp_DREB)
        orb_pct = calculate_orb_pct(oreb_team=10, dreb_opp=30)
        assert orb_pct == pytest.approx(0.25, rel=1e-3)

        # 4. DRB% = DREB / (DREB + Opp_OREB)
        drb_pct = calculate_drb_pct(dreb_team=35, oreb_opp=5)
        assert drb_pct == pytest.approx(0.875, rel=1e-3)

        # 5. FT Rate = FTM / FGA
        ft_rate = calculate_ft_rate(ftm=16, fga=80)
        assert ft_rate == pytest.approx(0.20, rel=1e-3)


class TestTemporalFeaturesAndFatigue:
    def test_rest_days_and_is_b2b(self):
        """Valida la correcta deducción de días de descanso e indicador back-to-back."""
        builder = FeatureBuilder(halflife=12.0)
        df = pd.DataFrame({
            "team_id": [1, 1, 1, 1],
            "game_id": ["G1", "G2", "G3", "G4"],
            "game_date": ["2023-11-01", "2023-11-02", "2023-11-05", "2023-11-06"],
            "pts": [100, 105, 110, 95],
            "opp_pts": [98, 102, 108, 99],
            "fga": [85, 86, 88, 84],
            "fta": [20, 22, 18, 25],
            "oreb": [10, 8, 9, 11],
            "tov": [12, 14, 11, 15],
            "opp_fga": [84, 85, 87, 85],
            "opp_fta": [22, 20, 19, 21],
            "opp_oreb": [8, 9, 10, 8],
            "opp_tov": [13, 12, 14, 11],
            "fgm": [40, 42, 44, 38],
            "fg3m": [10, 12, 11, 9],
            "ftm": [10, 9, 11, 10],
            "dreb": [32, 34, 33, 31],
            "opp_dreb": [30, 31, 32, 33],
            "opp_fgm": [39, 40, 43, 40],
            "opp_fg3m": [11, 10, 12, 9],
            "opp_ftm": [9, 12, 10, 10],
        })
        metrics_df = builder.compute_single_game_metrics(df)
        temporal_df = builder.compute_temporal_and_fatigue_features(metrics_df)

        # Juego 1: Primer partido (descanso imputado a 7, no b2b)
        assert temporal_df.loc[0, "rest_days"] == 7.0
        assert temporal_df.loc[0, "is_b2b"] == 0

        # Juego 2: 2023-11-02 vs 2023-11-01 (1 día de diferencia -> B2B)
        assert temporal_df.loc[1, "rest_days"] == 1.0
        assert temporal_df.loc[1, "is_b2b"] == 1

        # Juego 3: 2023-11-05 vs 2023-11-02 (3 días de diferencia -> no B2B)
        assert temporal_df.loc[2, "rest_days"] == 3.0
        assert temporal_df.loc[2, "is_b2b"] == 0

        # Juego 4: 2023-11-06 vs 2023-11-05 (1 día de diferencia -> B2B)
        assert temporal_df.loc[3, "rest_days"] == 1.0
        assert temporal_df.loc[3, "is_b2b"] == 1


class TestZeroDataLeakage:
    def test_strict_zero_data_leakage(self):
        """
        Demuestra matemáticamente la ausencia de fuga de datos en EWMA:
        Cualquier cambio en las estadísticas del partido N NO altera en absoluto
        las características EWMA disponibles previamente para predecir el partido N.
        """
        builder = FeatureBuilder(halflife=10.0)

        # Construir 6 partidos cronológicos
        dates = [f"2023-11-{d:02d}" for d in range(1, 7)]
        base_pts = [100, 105, 110, 95, 102, 108]

        def make_df(last_game_pts: int):
            pts_list = base_pts.copy()
            pts_list[-1] = last_game_pts
            df = pd.DataFrame({
                "team_id": [1] * 6,
                "game_id": [f"G{i}" for i in range(1, 7)],
                "game_date": dates,
                "pts": pts_list,
                "opp_pts": [95, 100, 105, 90, 98, 102],
                "fga": [85] * 6,
                "fta": [20] * 6,
                "oreb": [10] * 6,
                "tov": [12] * 6,
                "opp_fga": [85] * 6,
                "opp_fta": [20] * 6,
                "opp_oreb": [10] * 6,
                "opp_tov": [12] * 6,
                "fgm": [40] * 6,
                "fg3m": [10] * 6,
                "ftm": [10] * 6,
                "dreb": [30] * 6,
                "opp_dreb": [30] * 6,
                "opp_fgm": [38] * 6,
                "opp_fg3m": [9] * 6,
                "opp_ftm": [10] * 6,
            })
            metrics = builder.compute_single_game_metrics(df)
            return builder.compute_temporal_and_fatigue_features(metrics)

        # Ejecución normal: último partido con 108 puntos
        df_normal = make_df(last_game_pts=108)

        # Ejecución alterada: último partido con anomalía de 300 puntos
        df_altered = make_df(last_game_pts=300)

        # 1. El primer partido (índice 0) NO debe tener información previa (debe ser NaN)
        assert np.isnan(df_normal.loc[0, "ewma_pts"])

        # 2. Las features pre-partido para predecir el partido 6 (índice 5)
        # DEBEN SER IDÉNTICAS en ambas ejecuciones, a pesar de que en la segunda
        # el partido 6 anotó 300 puntos.
        feature_normal_g6 = df_normal.loc[5, "ewma_pts"]
        feature_altered_g6 = df_altered.loc[5, "ewma_pts"]

        assert feature_normal_g6 == pytest.approx(feature_altered_g6, abs=1e-7)
        # Y debe depender únicamente de los partidos 1 a 5
        assert not np.isnan(feature_normal_g6)
        assert feature_normal_g6 < 115.0  # El valor de 300 NO contaminó la feature


class TestMatchupDatasetAssembly:
    def test_build_matchup_dataset(self):
        """Verifica que el dataset de partidos contenga targets y variables diferenciales."""
        builder = FeatureBuilder(halflife=12.0, min_games=1)

        # Crear dos partidos con equipos LAL (1) y BOS (2)
        # Partido 1: LAL (Home) vs BOS (Away)
        # Partido 2: BOS (Home) vs LAL (Away)
        team_rows = pd.DataFrame([
            # Partido 1 - Local (LAL)
            {
                "game_id": "G01", "season": "2023-24", "game_date": "2023-11-01", "season_type": "Regular Season",
                "home_team_id": 1, "away_team_id": 2, "home_score": 110, "away_score": 105,
                "team_id": 1, "is_home": 1, "game_num": 1, "rest_days": 3.0, "is_b2b": 0,
                **{f"ewma_{m}": 105.0 for m in CANONICAL_METRICS}
            },
            # Partido 1 - Visitante (BOS)
            {
                "game_id": "G01", "season": "2023-24", "game_date": "2023-11-01", "season_type": "Regular Season",
                "home_team_id": 1, "away_team_id": 2, "home_score": 110, "away_score": 105,
                "team_id": 2, "is_home": 0, "game_num": 1, "rest_days": 1.0, "is_b2b": 1,
                **{f"ewma_{m}": 100.0 for m in CANONICAL_METRICS}
            },
        ])

        dataset = builder.build_matchup_dataset(team_rows)

        assert len(dataset) == 1
        row = dataset.iloc[0]

        # Targets
        assert row["point_spread"] == 5  # 110 - 105
        assert row["home_win"] == 1
        assert row["total_points"] == 215

        # Diferencial de descanso: home (3.0) - away (1.0) = +2.0
        assert row["rest_advantage"] == 2.0

        # Diferenciales Home - Away para métricas
        assert row["diff_pts"] == 5.0  # 105.0 - 100.0
        assert row["diff_ortg"] == 5.0

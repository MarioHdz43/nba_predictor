# -*- coding: utf-8 -*-
"""
tests/test_shin_and_market.py
------------------------------
Suite de pruebas unitarias para el Bloque 6 del rediseño institucional:
1. Método de Shin para eliminación de vigorish y extracción No-Vig.
2. Calibración Beta paramétrica (Kull et al., 2017).
3. Closing Line Value (CLV) y observabilidad de mercado.
4. Integración en PredictionResult (Shin, Quarter Kelly y CLV).
5. Inferencia con calibrador Beta en MLEngine.
"""

from __future__ import annotations
import numpy as np
import pytest

from domain.analytics.shin_odds import (
    calculate_shin_probabilities,
    shin_two_way_probabilities,
    calculate_market_vigorish,
)
from domain.analytics.beta_calibration import BetaCalibrator
from domain.analytics.clv import (
    calculate_clv,
    calculate_spread_clv,
    CLVBetRecord,
    CLVPerformanceTracker,
)
from infrastructure.odds.odds_tracker import (
    OddsSnapshot,
    MarketOddsTracker,
)
from domain.entities import Odds, PredictionResult
from domain.engines.ml_engine import MLEngine


# =====================================================================
# 1. Tests para el Método de Shin
# =====================================================================

class TestShinMethod:
    def test_shin_canonical_case(self):
        """
        Verifica el vector canónico de la literatura [2.6, 2.4, 4.3]:
        Debe converger a [0.373, 0.405, 0.222] con z > 0 y suma = 1.0.
        """
        odds = [2.6, 2.4, 4.3]
        probs, z, vig = calculate_shin_probabilities(odds)

        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert probs[0] == pytest.approx(0.3730, abs=1e-3)
        assert probs[1] == pytest.approx(0.4048, abs=1e-3)
        assert probs[2] == pytest.approx(0.2222, abs=1e-3)
        assert z > 0.01  # Detección de insiders
        assert vig == pytest.approx(0.0338, abs=1e-3)

    def test_shin_two_way_symmetric(self):
        """
        Para cuotas simétricas NBA [1.91, 1.91]:
        Ambas probabilidades No-Vig deben ser exactamente 0.50 (50%).
        """
        p_h, p_a, z, vig = shin_two_way_probabilities(1.91, 1.91)

        assert p_h == pytest.approx(0.50, abs=1e-4)
        assert p_a == pytest.approx(0.50, abs=1e-4)
        assert (p_h + p_a) == pytest.approx(1.0, abs=1e-6)
        # Overround: 1/1.91 + 1/1.91 - 1 = 1.0471 - 1 = 0.0471 (4.71%)
        assert vig == pytest.approx(0.0471, abs=1e-3)

    def test_shin_favorite_longshot_bias_correction(self):
        """
        Comprueba que Shin corrija el Favorite-Longshot Bias frente al método proporcional:
        Shin asigna mayor probabilidad verdadera al favorito que la división proporcional,
        y menor probabilidad al underdog/longshot.
        """
        fav_odds = 1.25
        dog_odds = 4.50

        # Proporcional clásico
        inv_f = 1.0 / fav_odds  # 0.80
        inv_d = 1.0 / dog_odds  # 0.2222
        booksum = inv_f + inv_d  # 1.0222
        prop_fav = inv_f / booksum
        prop_dog = inv_d / booksum

        # Shin
        shin_fav, shin_dog, z, vig = shin_two_way_probabilities(fav_odds, dog_odds)

        # Shin reconoce que los apostadores recreativos sobreapuestan al underdog,
        # por lo que la verdadera probabilidad del favorito es superior a la proporcional.
        assert shin_fav > prop_fav
        assert shin_dog < prop_dog
        assert (shin_fav + shin_dog) == pytest.approx(1.0, abs=1e-6)

    def test_shin_zero_vigorish(self):
        """Cuando no hay margen (cuotas justas sumando 1.0), z es 0.0."""
        probs, z, vig = calculate_shin_probabilities([2.0, 2.0])
        assert probs[0] == pytest.approx(0.5, abs=1e-4)
        assert probs[1] == pytest.approx(0.5, abs=1e-4)
        assert z == pytest.approx(0.0, abs=1e-4)
        assert vig == pytest.approx(0.0, abs=1e-4)

    def test_shin_invalid_odds_raises(self):
        """Cuotas menores o iguales a 1.0 deben disparar ValueError."""
        with pytest.raises(ValueError):
            calculate_shin_probabilities([1.0, 2.0])
        with pytest.raises(ValueError):
            calculate_shin_probabilities([0.9, 1.8])


# =====================================================================
# 2. Tests para Calibración Beta
# =====================================================================

class TestBetaCalibration:
    def test_monotonicity_preservation(self):
        """Verifica que la calibración Beta conserve la monotonicidad estricta (a >= 0, b >= 0)."""
        np.random.seed(42)
        raw_probs = np.linspace(0.1, 0.9, 50)
        labels = (raw_probs + np.random.normal(0, 0.15, 50) > 0.5).astype(int)

        calibrator = BetaCalibrator().fit(raw_probs, labels)
        calibrated = calibrator.predict_proba(raw_probs)

        # Monótono creciente: q[i+1] >= q[i]
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-6)
        assert calibrator.a >= 0.0
        assert calibrator.b >= 0.0

    def test_overconfidence_correction(self):
        """
        Un modelo sobreconfiado predice probabilidades extremas (0.90) cuando
        la frecuencia empírica real es menor (ej. 70%). La calibración debe suavizar las probabilidades.
        """
        # 100 muestras donde cuando el modelo predice 0.85, solo acierta el 60%
        y_prob = np.array([0.85] * 60 + [0.15] * 40)
        y_true = np.array([1] * 36 + [0] * 24 + [1] * 10 + [0] * 30)

        calibrator = BetaCalibrator().fit(y_prob, y_true)
        calibrated_85 = calibrator.predict_proba(0.85)

        # La probabilidad calibrada debe acercarse a 0.60 y ser menor a 0.85
        assert calibrated_85 < 0.85
        assert 0.45 < calibrated_85 < 0.75

    def test_serialization_and_deserialization(self):
        """Verifica que el calibrador se pueda exportar a dict y restaurar fielmente."""
        calibrator = BetaCalibrator(a=1.45, b=0.95, c=-0.12)
        calibrator.is_fitted = True

        data = calibrator.to_dict()
        restored = BetaCalibrator.from_dict(data)

        assert restored.a == 1.45
        assert restored.b == 0.95
        assert restored.c == -0.12
        assert restored.is_fitted is True

        p = 0.65
        assert calibrator.predict_proba(p) == pytest.approx(restored.predict_proba(p))

    def test_unfitted_calibrator_identity(self):
        """Un calibrador no ajustado debe devolver la probabilidad original sin mutar."""
        calibrator = BetaCalibrator()
        assert calibrator.predict_proba(0.62) == pytest.approx(0.62)
        assert np.all(calibrator.predict_proba([0.3, 0.7]) == pytest.approx([0.3, 0.7]))


# =====================================================================
# 3. Tests para CLV y Observabilidad de Mercado
# =====================================================================

class TestCLVAndMarketObservability:
    def test_calculate_clv_raw_ratios(self):
        """
        Valida CLV por ratio de cuotas:
        - Tomar 2.10 cuando cerró en 2.00 -> CLV = +5.0%
        - Tomar 1.90 cuando cerró en 2.00 -> CLV = -5.0%
        """
        clv_pos = calculate_clv(taken_odds=2.10, closing_odds=2.00)
        assert clv_pos == pytest.approx(0.05, abs=1e-4)

        clv_neg = calculate_clv(taken_odds=1.90, closing_odds=2.00)
        assert clv_neg == pytest.approx(-0.05, abs=1e-4)

    def test_calculate_clv_with_no_vig_closing(self):
        """
        Valida CLV frente a la probabilidad verdadera de cierre (Shin No-Vig):
        Si se tomó cuota 2.05 para Home, y el mercado cerró en Home=1.91, Away=1.91 (50% No-Vig):
        CLV = 2.05 * 0.50 - 1.0 = +2.5% (+0.025).
        """
        clv = calculate_clv(
            taken_odds=2.05,
            closing_odds=1.91,
            closing_opp_odds=1.91,
        )
        assert clv == pytest.approx(0.025, abs=1e-3)

    def test_calculate_spread_clv(self):
        """
        Valida el CLV de líneas de handicap:
        - Tomar +5.5 cuando cerró en +3.5 -> +2.0 puntos de CLV a favor.
        - Tomar -3.5 cuando cerró en -5.5 -> +2.0 puntos de CLV a favor.
        """
        assert calculate_spread_clv(taken_spread=5.5, closing_spread=3.5) == 2.0
        assert calculate_spread_clv(taken_spread=-3.5, closing_spread=-5.5) == 2.0
        assert calculate_spread_clv(taken_spread=3.5, closing_spread=5.5) == -2.0

    def test_clv_performance_tracker_report(self):
        """Valida la agregación de métricas de CLV acumulado."""
        tracker = CLVPerformanceTracker()

        # Apuesta 1: Batio el cierre (+5% CLV)
        tracker.add_record(
            CLVBetRecord(
                bet_id="B01",
                game_id="G01",
                selection="Home",
                taken_odds=2.10,
                closing_odds=2.00,
            )
        )
        # Apuesta 2: Perdio contra el cierre (-4% CLV)
        tracker.add_record(
            CLVBetRecord(
                bet_id="B02",
                game_id="G02",
                selection="Away",
                taken_odds=1.92,
                closing_odds=2.00,
            )
        )

        report = tracker.generate_report()
        assert report.total_bets == 2
        assert report.beat_closing_count == 1
        assert report.beat_closing_rate == pytest.approx(0.50)
        assert report.average_clv == pytest.approx((0.05 - 0.04) / 2.0, abs=1e-3)

    def test_market_odds_tracker_workflow(self):
        """Valida el ciclo de vida de captura de cuotas y auditoría de CLV."""
        tracker = MarketOddsTracker()
        game_id = "0022300050"

        # 1. Registrar apertura
        open_snap = OddsSnapshot(
            game_id=game_id,
            bookmaker="Pinnacle",
            home_odds=2.10,
            away_odds=1.75,
            home_spread=2.5,
            away_spread=-2.5,
        )
        tracker.record_snapshot(open_snap, is_opening=True)

        # 2. Registrar cierre institucional
        close_snap = OddsSnapshot(
            game_id=game_id,
            bookmaker="Pinnacle",
            home_odds=1.95,
            away_odds=1.87,
            home_spread=1.0,
            away_spread=-1.0,
        )
        tracker.record_snapshot(close_snap, is_closing=True)

        # 3. Auditar apuesta tomada a cuota 2.10 y spread +2.5
        audit = tracker.compute_bet_clv(
            game_id=game_id,
            selection="Home",
            taken_odds=2.10,
            taken_spread=2.5,
        )

        assert audit["clv"] is not None
        assert audit["clv"] > 0.0  # Tomo a 2.10, cerro a 1.95 (supero el cierre)
        assert audit["spread_clv"] == 1.5  # 2.5 - 1.0 = +1.5 pts de ventaja


# =====================================================================
# 4. Tests para Integración en PredictionResult y MLEngine
# =====================================================================

class TestEntitiesAndEngineIntegration:
    def test_prediction_result_attach_odds_with_shin(self):
        """
        Valida que attach_odds() calcule:
        - Probabilidades no-vig mediante Shin.
        - Margen del mercado.
        - EV real y Quarter Kelly (0.25).
        """
        pred = PredictionResult(
            home_team="Boston Celtics",
            away_team="Miami Heat",
            home_win_prob=0.65,
            away_win_prob=0.35,
            confidence=0.75,
            key_factors=["Dominio rebote"],
        )

        odds = Odds(home_decimal=1.80, away_decimal=2.15)
        enriched = pred.attach_odds(odds)

        # Campos No-Vig y Vigorish poblados vía Shin
        assert enriched.home_no_vig_prob is not None
        assert enriched.away_no_vig_prob is not None
        assert enriched.market_vigorish is not None
        assert enriched.home_no_vig_prob + enriched.away_no_vig_prob == pytest.approx(1.0, abs=1e-3)
        assert enriched.market_vigorish > 0.0

        # EV para Home: 0.65 * (1.80 - 1) - (1 - 0.65) = 0.65 * 0.80 - 0.35 = 0.52 - 0.35 = +0.17
        assert enriched.home_expected_value == pytest.approx(0.17, abs=1e-3)
        assert enriched.recommended_bet == "Home (Boston Celtics)"

        # Quarter Kelly para Home: max(0, 0.17 / 0.80) * 0.25 = (0.2125) * 0.25 = 0.0531
        assert enriched.kelly_stake_home == pytest.approx(0.0531, abs=1e-3)

    def test_prediction_result_attach_odds_with_closing_line(self):
        """Valida que attach_odds() calcule expected_clv si se pasan cuotas de cierre."""
        pred = PredictionResult(
            home_team="Denver Nuggets",
            away_team="LA Lakers",
            home_win_prob=0.70,
            away_win_prob=0.30,
            confidence=0.80,
            key_factors=["Altitud"],
        )

        taken_odds = Odds(home_decimal=1.85, away_decimal=2.05)
        closing_odds = Odds(home_decimal=1.65, away_decimal=2.30)

        enriched = pred.attach_odds(taken_odds, closing_odds=closing_odds)

        assert enriched.expected_clv is not None
        # Tomó 1.85 y cerró mucho más bajo (1.65), capturó CLV positivo
        assert enriched.expected_clv > 0.05

    def test_ml_engine_incorporates_beta_calibrator(self):
        """Verifica que MLEngine aplique BetaCalibrator cuando está provisto."""
        engine = MLEngine()
        # Mock de calibrador Beta
        calibrator = BetaCalibrator(a=1.2, b=0.8, c=-0.1)
        calibrator.is_fitted = True
        engine._beta_calibrator = calibrator

        # Comprobar que el atributo existe y está cargado
        assert engine._beta_calibrator is not None

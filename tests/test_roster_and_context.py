# -*- coding: utf-8 -*-
"""
tests/test_roster_and_context.py
---------------------------------
Suite de pruebas unitarias para el Bloque 5 del rediseño institucional:
1. Parser del NBA Injury Report y probabilidades bayesianas a priori.
2. Proyección de rotaciones, conservación canónica de 240 minutos y degradación RAPM.
3. Contexto geográfico y viajes: Haversine, husos horarios, altitud y fatiga.
4. Integración de variables contextuales en FeatureBuilder.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

from infrastructure.scrapers.injury_parser import (
    InjuryStatus,
    PlayerInjuryReport,
    parse_injury_status,
    get_availability_probability,
    parse_injury_text_report,
    parse_injury_records,
)
from domain.analytics.roster_impact import (
    PlayerRosterProfile,
    RosterProjectionResult,
    RosterProjectionEngine,
)
from domain.analytics.travel_context import (
    calculate_haversine_distance,
    calculate_travel_miles,
    calculate_time_zones_crossed,
    is_altitude_advantage,
    calculate_travel_fatigue_index,
    NBA_TEAM_COORDINATES,
)
from domain.analytics.feature_builder import FeatureBuilder


# =====================================================================
# 1. Tests para Injury Parser y Probabilidades Bayesianas
# =====================================================================

class TestInjuryParser:
    def test_bayesian_availability_probabilities(self):
        """Verifica las probabilidades bayesianas a priori para cada estatus de lesión."""
        assert get_availability_probability(InjuryStatus.OUT) == 0.0
        assert get_availability_probability(InjuryStatus.DOUBTFUL) == 0.10
        assert get_availability_probability(InjuryStatus.QUESTIONABLE) == 0.50
        assert get_availability_probability(InjuryStatus.PROBABLE) == 0.90
        assert get_availability_probability(InjuryStatus.AVAILABLE) == 1.00

    def test_parse_injury_status_strings(self):
        """Verifica la normalización de cadenas de texto a InjuryStatus enum."""
        assert parse_injury_status("Out") == InjuryStatus.OUT
        assert parse_injury_status("DOUBTFUL") == InjuryStatus.DOUBTFUL
        assert parse_injury_status("Questionable - Right Knee Soreness") == InjuryStatus.QUESTIONABLE
        assert parse_injury_status("Probable") == InjuryStatus.PROBABLE
        assert parse_injury_status("Available") == InjuryStatus.AVAILABLE
        assert parse_injury_status("Active") == InjuryStatus.AVAILABLE
        assert parse_injury_status("Desconocido") == InjuryStatus.QUESTIONABLE

    def test_parse_injury_text_report(self):
        """Valida el parseo de un reporte en formato texto oficial de la NBA."""
        sample_report = """
        Los Angeles Lakers: LeBron James (Questionable - Left Ankle Peroneal Tendinopathy); Anthony Davis (Probable - Achilles)
        Golden State Warriors: Stephen Curry (Available); Draymond Green (Out - Back)
        """
        reports = parse_injury_text_report(sample_report)
        assert len(reports) == 4

        lbj = next(r for r in reports if "LeBron James" in r.player_name)
        assert lbj.team_name == "Los Angeles Lakers"
        assert lbj.status == InjuryStatus.QUESTIONABLE
        assert lbj.expected_availability == 0.50

        curry = next(r for r in reports if "Stephen Curry" in r.player_name)
        assert curry.team_name == "Golden State Warriors"
        assert curry.status == InjuryStatus.AVAILABLE
        assert curry.expected_availability == 1.00

        green = next(r for r in reports if "Draymond Green" in r.player_name)
        assert green.status == InjuryStatus.OUT
        assert green.expected_availability == 0.00

    def test_parse_injury_records_list(self):
        """Valida el parseo de registros estructurados (JSON/dict)."""
        records = [
            {"player_name": "Nikola Jokic", "team": "DEN", "status": "Available", "reason": "Rest"},
            {"player_name": "Jamal Murray", "team": "DEN", "status": "Doubtful", "reason": "Hamstring"},
        ]
        parsed = parse_injury_records(records)
        assert len(parsed) == 2
        assert parsed[0].player_name == "Nikola Jokic"
        assert parsed[0].expected_availability == 1.00
        assert parsed[1].player_name == "Jamal Murray"
        assert parsed[1].expected_availability == 0.10


# =====================================================================
# 2. Tests para Proyección de Rotaciones y RAPM
# =====================================================================

class TestRosterProjectionEngine:
    @pytest.fixture
    def sample_roster(self) -> list[PlayerRosterProfile]:
        """Crea un roster balanceado de 8 jugadores sumando exactamente 240 minutos."""
        return [
            PlayerRosterProfile(player_name="Superstar 1", team_id=1, baseline_minutes=36.0, o_rapm=4.5, d_rapm=-1.5),
            PlayerRosterProfile(player_name="Star 2", team_id=1, baseline_minutes=34.0, o_rapm=2.5, d_rapm=1.0),
            PlayerRosterProfile(player_name="Starter 3", team_id=1, baseline_minutes=32.0, o_rapm=1.0, d_rapm=-0.5),
            PlayerRosterProfile(player_name="Starter 4", team_id=1, baseline_minutes=30.0, o_rapm=-0.5, d_rapm=0.5),
            PlayerRosterProfile(player_name="Starter 5", team_id=1, baseline_minutes=28.0, o_rapm=0.0, d_rapm=0.0),
            PlayerRosterProfile(player_name="Sixth Man", team_id=1, baseline_minutes=26.0, o_rapm=1.0, d_rapm=0.5),
            PlayerRosterProfile(player_name="Bench 7", team_id=1, baseline_minutes=28.0, o_rapm=-1.0, d_rapm=0.0),
            PlayerRosterProfile(player_name="Bench 8", team_id=1, baseline_minutes=26.0, o_rapm=-1.5, d_rapm=1.0),
        ]

    def test_baseline_preserves_240_minutes(self, sample_roster):
        """Verifica que el roster proyectado conserve exactamente 240 minutos totales."""
        engine = RosterProjectionEngine()
        result = engine.calculate_team_projected_ratings(sample_roster)

        assert sum(p.baseline_minutes for p in sample_roster) == pytest.approx(240.0)
        assert result.total_projected_minutes == pytest.approx(240.0)
        assert result.minutes_lost == pytest.approx(0.0)
        assert result.net_rating_degradation == pytest.approx(0.0)

    def test_star_injury_causes_rating_degradation(self, sample_roster):
        """
        Verifica que si la superestrella (o_rapm=+4.5, d_rapm=-1.5) queda OUT:
        1. Se conservan los 240 minutos mediante redistribución a nivel de reemplazo.
        2. El Net Rating se degrada significativamente.
        3. minutes_lost refleja los 36 minutos del jugador ausente.
        """
        engine = RosterProjectionEngine()
        baseline_result = engine.calculate_team_projected_ratings(sample_roster)

        # Lesionar a la superestrella
        injured_roster = [p for p in sample_roster]
        injured_roster[0] = PlayerRosterProfile(
            player_name="Superstar 1",
            team_id=1,
            baseline_minutes=36.0,
            o_rapm=4.5,
            d_rapm=-1.5,
            injury_status=InjuryStatus.OUT,
            availability_prob=0.0,
        )

        injured_result = engine.calculate_team_projected_ratings(injured_roster)

        # 240 minutos exactos conservados
        assert injured_result.total_projected_minutes == pytest.approx(240.0)
        assert injured_result.minutes_lost == pytest.approx(36.0)

        # Degradación negativa del Net Rating: el equipo pierde a su principal generador
        assert injured_result.projected_net_rating < baseline_result.projected_net_rating
        assert injured_result.net_rating_degradation < -3.0
        # Absorción de minutos por rotación secundaria y jugador de reemplazo
        assert injured_result.net_rating_degradation == pytest.approx(-5.58, abs=0.1)

    def test_questionable_status_partial_impact(self, sample_roster):
        """Verifica que un estatus QUESTIONABLE (50% disponibilidad) cause un impacto proporcional."""
        engine = RosterProjectionEngine()
        baseline_result = engine.calculate_team_projected_ratings(sample_roster)

        q_roster = [p for p in sample_roster]
        q_roster[0] = PlayerRosterProfile(
            player_name="Superstar 1",
            team_id=1,
            baseline_minutes=36.0,
            o_rapm=4.5,
            d_rapm=-1.5,
            injury_status=InjuryStatus.QUESTIONABLE,
            availability_prob=0.50,
        )

        q_result = engine.calculate_team_projected_ratings(q_roster)

        assert q_result.minutes_lost == pytest.approx(18.0)
        assert q_result.total_projected_minutes == pytest.approx(240.0)
        # El impacto debe ser la mitad que cuando está OUT (36 min vs 18 min perdidos)
        assert q_result.net_rating_degradation < 0.0
        assert q_result.projected_net_rating < baseline_result.projected_net_rating


# =====================================================================
# 3. Tests para Contexto Geográfico, Viajes y Altitud
# =====================================================================

class TestTravelContext:
    def test_haversine_distance_accuracy(self):
        """Valida que la distancia Haversine entre Los Ángeles y Boston sea ~2600 millas."""
        lal = NBA_TEAM_COORDINATES["LAL"]
        bos = NBA_TEAM_COORDINATES["BOS"]

        dist = calculate_haversine_distance(lal.latitude, lal.longitude, bos.latitude, bos.longitude)
        assert 2580.0 <= dist <= 2630.0

    def test_calculate_travel_miles_same_team(self):
        """Si un equipo juega en casa o contra sí mismo, la distancia es 0."""
        assert calculate_travel_miles("LAL", "LAL") == 0.0
        assert calculate_travel_miles(1610612747, 1610612747) == 0.0

    def test_calculate_travel_miles_lookup(self):
        """Valida cálculo de millas usando abreviaturas e IDs oficiales de la NBA."""
        # LAL (1610612747) a BOS (1610612738)
        dist_abbr = calculate_travel_miles("LAL", "BOS")
        dist_ids = calculate_travel_miles(1610612747, 1610612738)
        assert dist_abbr == pytest.approx(dist_ids, rel=1e-3)
        assert 2580.0 <= dist_abbr <= 2630.0

    def test_calculate_time_zones_crossed(self):
        """Valida husos horarios cruzados entre costas de EE.UU."""
        # Pacífico a Este: UTC-8 a UTC-5 = 3 husos
        assert calculate_time_zones_crossed("LAL", "BOS") == 3
        assert calculate_time_zones_crossed("BOS", "LAL") == 3
        # Dentro de la misma zona horaria
        assert calculate_time_zones_crossed("NYK", "BOS") == 0
        assert calculate_time_zones_crossed("LAL", "GSW") == 0

    def test_altitude_advantage(self):
        """Valida que únicamente Denver y Utah tengan ventaja de altitud (> 4000 pies)."""
        # Denver y Utah: True
        assert is_altitude_advantage("DEN") is True
        assert is_altitude_advantage("UTA") is True
        assert is_altitude_advantage(1610612743) is True  # Denver Nuggets ID
        assert is_altitude_advantage(1610612762) is True  # Utah Jazz ID

        # Equipos a nivel del mar o baja altitud: False
        assert is_altitude_advantage("LAL") is False
        assert is_altitude_advantage("MIA") is False
        assert is_altitude_advantage("BOS") is False
        assert is_altitude_advantage("GSW") is False
        assert is_altitude_advantage(1610612747) is False  # Lakers ID

    def test_travel_fatigue_index_properties(self):
        """Valida las propiedades del índice compuesto de fatiga por viaje."""
        # Viaje corto con descanso abundante: fatiga mínima
        fatigue_low = calculate_travel_fatigue_index(travel_miles=300, time_zones=0, rest_days=3, is_b2b=0)
        # Viaje largo cross-country en back-to-back cruzando 3 zonas horarias: fatiga alta
        fatigue_high = calculate_travel_fatigue_index(travel_miles=2600, time_zones=3, rest_days=0, is_b2b=1)

        assert fatigue_low < fatigue_high
        assert fatigue_low >= 0.0
        assert fatigue_high > 1.5


# =====================================================================
# 4. Tests para Integración de Contexto en FeatureBuilder
# =====================================================================

class TestFeatureBuilderContextIntegration:
    def test_build_matchup_dataset_includes_context_features(self):
        """
        Verifica que build_matchup_dataset agregue las 4 nuevas columnas contextuales:
        - is_altitude_advantage
        - travel_miles_away
        - time_zones_crossed_away
        - travel_fatigue_away
        """
        fb = FeatureBuilder(min_games=1)

        # Generar DataFrame simulado con métricas calculadas para Home (DEN) y Away (MIA)
        # Denver ID: 1610612743, Miami ID: 1610612748
        data = {
            "game_id": ["G001", "G001"],
            "season": ["2023-24", "2023-24"],
            "game_date": ["2024-01-10", "2024-01-10"],
            "season_type": ["Regular Season", "Regular Season"],
            "team_id": [1610612743, 1610612748],
            "home_team_id": [1610612743, 1610612743],
            "away_team_id": [1610612748, 1610612748],
            "is_home": [1, 0],
            "home_score": [115, 115],
            "away_score": [105, 105],
            "game_num": [10, 10],
            "rest_days": [2, 0],
            "is_b2b": [0, 1],
        }

        # Añadir EWMA de métricas canónicas
        from domain.analytics.feature_builder import CANONICAL_METRICS
        for m in CANONICAL_METRICS:
            data[f"ewma_{m}"] = [110.0, 105.0]

        team_metrics_df = pd.DataFrame(data)
        matchups = fb.build_matchup_dataset(team_metrics_df)

        assert not matchups.empty
        # Verificar que existan las 4 columnas
        assert "is_altitude_advantage" in matchups.columns
        assert "travel_miles_away" in matchups.columns
        assert "time_zones_crossed_away" in matchups.columns
        assert "travel_fatigue_away" in matchups.columns

        row = matchups.iloc[0]
        # Denver es local -> is_altitude_advantage = 1
        assert row["is_altitude_advantage"] == 1
        # Miami viajando a Denver -> ~1700 millas
        assert row["travel_miles_away"] > 1500.0
        # Miami (UTC-5) a Denver (UTC-7) -> 2 husos
        assert row["time_zones_crossed_away"] == 2
        # Miami en B2B tras viaje largo -> fatiga positiva
        assert row["travel_fatigue_away"] > 1.0

# -*- coding: utf-8 -*-
"""
domain/entities.py
------------------
Entidades y Value Objects del dominio. Sin dependencias de infraestructura.
Estas clases representan los conceptos centrales del negocio.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Value Objects (inmutables, igualdad por valor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamStats:
    """Estadísticas agregadas de un equipo para una temporada."""
    pts: float = 0.0
    fg_pct: float = 0.0
    fg3_pct: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    tov: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    plus_minus: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "PTS": self.pts,
            "FG_PCT": self.fg_pct,
            "FG3_PCT": self.fg3_pct,
            "REB": self.reb,
            "AST": self.ast,
            "TOV": self.tov,
            "STL": self.stl,
            "BLK": self.blk,
            "PLUS_MINUS": self.plus_minus,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "TeamStats":
        return cls(
            pts=float(d.get("PTS", 0)),
            fg_pct=float(d.get("FG_PCT", 0)),
            fg3_pct=float(d.get("FG3_PCT", 0)),
            reb=float(d.get("REB", 0)),
            ast=float(d.get("AST", 0)),
            tov=float(d.get("TOV", 0)),
            stl=float(d.get("STL", 0)),
            blk=float(d.get("BLK", 0)),
            plus_minus=float(d.get("PLUS_MINUS", 0)),
        )


@dataclass(frozen=True)
class TeamForm:
    """
    Forma reciente del equipo (últimos N partidos).
    Usada para dar más peso a la racha actual vs promedio de temporada.
    """
    wins: int = 0
    losses: int = 0
    points_per_game: float = 0.0
    opponent_points_per_game: float = 0.0
    last_games: int = 5

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def net_rtg(self) -> float:
        return self.points_per_game - self.opponent_points_per_game

    @property
    def is_hot(self) -> bool:
        return self.win_pct >= 0.6

    @property
    def is_cold(self) -> bool:
        return self.win_pct <= 0.4


@dataclass(frozen=True)
class InjuredPlayer:
    """Jugador lesionado con impacto estimado."""
    name: str
    impact: float = 0.0
    position: str = "unknown"

    @property
    def is_starter(self) -> bool:
        return self.position in ("PG", "SG", "SF", "PF", "C")


@dataclass(frozen=True)
class H2HStats:
    """
    Historial head-to-head entre dos equipos.
    Usado para ajustar predicción basado en ventaja histórica.
    """
    home_team: str
    away_team: str
    home_wins: int = 0
    away_wins: int = 0
    total_games: int = 0

    @property
    def home_win_pct(self) -> float:
        if self.total_games == 0:
            return 0.5
        return self.home_wins / self.total_games

    @property
    def advantage(self) -> float:
        return self.home_win_pct - 0.5

    @property
    def sample_size_ok(self) -> bool:
        return self.total_games >= 3


@dataclass(frozen=True)
class PlayerStat:
    name: str
    avg_points: float
    avg_minutes: float


@dataclass(frozen=True)
class Odds:
    """
    Cuotas de un bookmaker para un partido.
    Necesarias para calcular Expected Value real.
    """
    home_decimal: float  # ej: 1.85
    away_decimal: float  # ej: 2.10

    def home_implied_prob(self) -> float:
        return 1.0 / self.home_decimal

    def away_implied_prob(self) -> float:
        return 1.0 / self.away_decimal

    def has_overround(self) -> bool:
        return (self.home_implied_prob() + self.away_implied_prob()) > 1.0


# ---------------------------------------------------------------------------
# Entidades (tienen identidad, pueden mutar con el tiempo)
# ---------------------------------------------------------------------------

@dataclass
class Team:
    """Equipo NBA con su identidad y datos de temporada."""
    team_id: int
    full_name: str
    abbreviation: str
    season: str
    stats: TeamStats = field(default_factory=TeamStats)
    top_players: List[PlayerStat] = field(default_factory=list)
    recent_form: Optional[TeamForm] = None
    injured_players: List[InjuredPlayer] = field(default_factory=list)
    fetched_at: Optional[datetime] = None
    source: str = "unknown"

    @property
    def injury_adjustment(self) -> float:
        total = sum(p.impact for p in self.injured_players if p.is_starter)
        return -total

    @property
    def has_key_injuries(self) -> bool:
        return any(p.is_starter for p in self.injured_players)

    def is_stale(self, ttl_hours: int = 2) -> bool:
        if self.fetched_at is None:
            return True
        elapsed = (datetime.utcnow() - self.fetched_at).total_seconds()
        return elapsed > ttl_hours * 3600


@dataclass
class PredictionResult:
    """
    Resultado de una predicción para un partido.
    Contiene probabilidades, confianza y valor de apuesta con cuotas reales.
    """
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    confidence: float
    key_factors: List[str]
    predicted_at: datetime = field(default_factory=datetime.utcnow)

    # Opcionales: solo disponibles si se proveen cuotas
    odds: Optional[Odds] = None
    home_expected_value: Optional[float] = None
    away_expected_value: Optional[float] = None
    recommended_bet: str = "No bet"
    kelly_stake_home: Optional[float] = None
    kelly_stake_away: Optional[float] = None
    id: Optional[int] = None
    actual_result: Optional[str] = None
    home_no_vig_prob: Optional[float] = None
    away_no_vig_prob: Optional[float] = None
    market_vigorish: Optional[float] = None
    expected_clv: Optional[float] = None
    projected_margin: Optional[float] = None

    def attach_odds(
        self,
        odds: Odds,
        closing_odds: Optional[Odds] = None,
    ) -> "PredictionResult":
        """
        Calcula el Expected Value real y el tamaño óptimo de apuesta
        según el Criterio de Kelly Fraccional (0.25 / Quarter Kelly),
        despojando el margen del corredor mediante el Método de Shin.

        EV = (prob_ganar × ganancia_neta) - (prob_perder × apuesta_unitaria)
        Para una apuesta de 1 unidad con ganancia neta b = odds - 1:
          EV_home = home_prob × b_home - (1 - home_prob)
          f*_home = max(0, (home_prob × b_home - (1 - home_prob)) / b_home) × 0.25
        """
        from domain.analytics.shin_odds import shin_two_way_probabilities
        from domain.analytics.clv import calculate_clv

        # 1. Extracción de probabilidades justas No-Vig mediante Shin
        home_nv, away_nv, _, vig = shin_two_way_probabilities(
            odds.home_decimal, odds.away_decimal
        )

        # 2. Expected Value y Quarter Kelly
        b_home = odds.home_decimal - 1.0
        ev_home = (
            self.home_win_prob * b_home
            - (1.0 - self.home_win_prob) * 1.0
        )
        kelly_home = 0.0
        if b_home > 0:
            raw_kelly_h = (self.home_win_prob * b_home - (1.0 - self.home_win_prob)) / b_home
            kelly_home = max(0.0, raw_kelly_h) * 0.25

        b_away = odds.away_decimal - 1.0
        ev_away = (
            self.away_win_prob * b_away
            - (1.0 - self.away_win_prob) * 1.0
        )
        kelly_away = 0.0
        if b_away > 0:
            raw_kelly_a = (self.away_win_prob * b_away - (1.0 - self.away_win_prob)) / b_away
            kelly_away = max(0.0, raw_kelly_a) * 0.25

        bet = "No bet"
        if ev_home > 0 and ev_home >= ev_away:
            bet = f"Home ({self.home_team})"
        elif ev_away > 0:
            bet = f"Away ({self.away_team})"

        # 3. Cálculo de CLV si se proporcionan cuotas de cierre institucionales
        exp_clv: Optional[float] = None
        if closing_odds is not None:
            if ev_home > 0 and ev_home >= ev_away:
                exp_clv = calculate_clv(
                    taken_odds=odds.home_decimal,
                    closing_odds=closing_odds.home_decimal,
                    closing_opp_odds=closing_odds.away_decimal,
                )
            elif ev_away > 0:
                exp_clv = calculate_clv(
                    taken_odds=odds.away_decimal,
                    closing_odds=closing_odds.away_decimal,
                    closing_opp_odds=closing_odds.home_decimal,
                )

        return PredictionResult(
            home_team=self.home_team,
            away_team=self.away_team,
            home_win_prob=self.home_win_prob,
            away_win_prob=self.away_win_prob,
            confidence=self.confidence,
            key_factors=self.key_factors,
            predicted_at=self.predicted_at,
            odds=odds,
            home_expected_value=round(ev_home, 4),
            away_expected_value=round(ev_away, 4),
            recommended_bet=bet,
            kelly_stake_home=round(kelly_home, 4),
            kelly_stake_away=round(kelly_away, 4),
            id=self.id,
            actual_result=self.actual_result,
            home_no_vig_prob=round(home_nv, 4),
            away_no_vig_prob=round(away_nv, 4),
            market_vigorish=round(vig, 4),
            expected_clv=exp_clv,
            projected_margin=self.projected_margin,
        )
